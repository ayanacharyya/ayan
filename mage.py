'''
Collection of python routines by Ayan acharyya applicable on mage sample, mainly to be used by the code EW_fitter.py
to fit gaussian profiles to spectral lines.
Started July 2016
'''
import os
HOME = os.getenv('HOME')+'/'
import sys
sys.path.append(HOME+'Dropbox/MagE_atlas/Tools')
sys.path.append(HOME+'Dropbox/MagE_atlas/Tools/Contrib')
import jrr
import splot_util as s
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1000)
from  astropy.io import ascii
from matplotlib import pyplot as plt
mage_mode = "released"
import astropy.convolution
import re
from scipy.optimize import curve_fit
from matplotlib.backends.backend_pdf import PdfPages
import subprocess
from operator import itemgetter
from astropy.stats import gaussian_fwhm_to_sigma as gf2s
from astropy.stats import gaussian_sigma_to_fwhm as gs2f
import argparse as ap
import statsmodels.robust.scale as stat
import scipy.stats as stats
from scipy.interpolate import interp1d
from scipy.interpolate import InterpolatedUnivariateSpline as extrapolate
from scipy import asarray as ar,exp

#-------Function for making new linelist files. DO NOT DELETE even if unused--------------------------
def makelist(linelist):
    #in the following list, keep (comment out rest) all the lines you want to be extracted from Rigby's .linelist files
    #into our new labframe.shortlinelist files
    target_line_labels_to_fit = [
     'C III 1247', \
     'Si III 1294', 'Si III 1296', 'C III 1296', 'Si III 1298', 'O I 1302', 'Si II 1304', \
     'Si II 1309', 'O I 1304', 'O I 1306', \
     'C III] 1906', 'C III] 1908', \
     'C II 1334', 'C II* 1335', 'C II 1323', 'N III 1324', 'O IV 1341', \
     #'O IV 1343', \
     'Si III 1417', \
     'S V 1501', \
     'C III 1620', \
     'Fe IV 1717',\
     'Fe III 1954', 'Fe III 1958', \
     'C II] 2326', \
     'He II 1640', 'O III] 1666', \
     'Si II 1533',\
     'C III 2297',\
     '[O III] 2321',\
     '[O III] 2331',\
     'O III] 1666',\
     'O III] 1660',\
     'N II 1084', 'N II 1085',\
     'N I 1134a', 'N I 1134b', 'N I 1134c',\
     'N III 1183', 'N III 1184', 'N IV 1169', 'N III 1324', 'N IV 1718',\
     'N IV] 1486', 'N II] 2140',\
     'Mn II 2576','Fe II 2586','Mn II 2594','Fe II 2599','Fe II 2600',\
    'Mn II 2606','Fe II 2607','Fe II 2612','Fe II 2614','Fe II 2618','Fe II 2621','Fe II 2622','Fe II 2626','Fe II 2629','Fe II 2631','Fe II 2632',\
    'Mg II 2796','Mg II 2803','Mg I 2852',\
    'Ti II 3073','Ti II 3230','Ti II 3239','Ti II 3242','Ti II 3384',\
    'Ca II 3934','Ca II 3969','Ca I 4227',\
    'Na I 5891','Na I 5897',\
    'Li I 6709'\
]
    (LL, zz_redundant) = jrr.mage.get_linelist(linelist)
    line_full = pd.DataFrame(columns=['LineID','restwave','type','source'])
    for label in target_line_labels_to_fit:
        try:
            t = LL[(LL['lab1']+' '+LL['lab2']).eq(label) & LL['type'].ne('INTERVE')].type.values[0]
        except IndexError:
            print label, 'not found'
            continue
        row = np.array([''+label.replace(' ', ''), LL[(LL['lab1']+' '+LL['lab2']).eq(label) & LL['type'].ne('INTERVE')].restwav.values[0], t, 'Leitherer'])
        line_full.loc[len(line_full)] = row
    fout = 'labframe.shortlinelist'
    line_full.to_csv(fout, sep='\t',mode ='a', index=None)

    fn = open('/Users/acharyya/Mappings/lab/targetlines.txt','r')
    fout2 = open(fout,'a')
    for lin in fn.readlines():
        if len(lin.split())>1 and lin.split()[0][0] != '#':
            fout2.write(''+lin.split()[1]+'   '+lin.split()[0]+'   EMISSION'+'   MAPPINGS\n')
    fout2.close()
    fn.close()
    LL = pd.read_table(fout, delim_whitespace=True, comment='#') # Load different, shorter linelist to fit lines
    LL = LL.sort('restwave')
    head = '#New customised linelist mostly only for emission lines\n\
#Columns are:\n\
LineID  restwave    type    source\n'

    np.savetxt(fout, np.transpose([LL.LineID, LL.restwave, LL.type, LL.source]), "%s   %s   %s   %s", \
    header=head, comments='')
    print 'Created new linelist', fout
    #subprocess.call(['python /Users/acharyya/Desktop/mage_plot/comment_file.py '+'/Users/acharyya/Dropbox/MagE_atlas/Tools/Contrib/'+fout],shell=True)
#-----------Function to flag skylines in a different way than by JRR, if required----------------------
def flag_skylines(sp) :
    # Mask skylines [O I] 5577\AA\ and [O I]~6300\AA,
    skyline = np.array([5577., 6300.])
    prev_skywidth = 17.0 #used by jrr.mage.flag_skylines
    skywidth = skyline*250./3e5 # flag spectrum +- skywidth of the skyline, skywidth different from JRR
                                # masking vel = 250 km/s on either side of skylines
    for pp in range(len(skyline)):
        sp.badmask.loc[sp['wave'].between(skyline[pp]-prev_skywidth, skyline[pp]+prev_skywidth)] = False #undo jrr.mage masking
        sp.badmask.loc[sp['wave'].between(skyline[pp]-skywidth[pp], skyline[pp]+skywidth[pp])] = True #redo new masking

#-------------Function to fit autocont using jrr.mage.auto_fit.cont------------------
def fit_autocont(sp_orig, zz_sys, line_path, filename):
    if 'stack' in filename:
        linelist = line_path+'stacked.linelist' #to provide line list for jrr.mage.auto_fit_cont to mask out regions of the spectra
    elif 'new-format' in filename:
        linelist = line_path+'stacked.linelist'
    elif 'esi' in filename:
        linelist = line_path+'stacked.linelist' 
    else:
        linelist = jrr.mage.get_linelist_name(filename, line_path)   # convenience function
    (LL, zz_redundant) = jrr.mage.get_linelist(linelist)  # Load the linelist to fit auto-cont    
    jrr.mage.auto_fit_cont(sp_orig, LL, zz_sys)  # Automatically fit continuum.  results written to sp.fnu_autocont, sp.flam_autocont.

#-------------Function to read in specra file of format (obswave, fnu, fnu_u, restwave), based on jrr.mage.open_spectrum------------------
def open_esi_spectrum(infile, getclean=True) :
    '''Reads a reduced ESI spectrum
      Inputs:   filename to read in
      Outputs:  the object spectrum, in both flam and fnu (why not?) all in Pandas data frame
      Pandas keys:  wave, fnu, fnu_u, flam, flam_u
      call:  (Pandas_spectrum_dataframe, spectral_resolution) = ayan.mage.open_spectrum(infile)
    '''
    sp =  pd.read_table(infile, delim_whitespace=True, comment="#", header=0)#, names=names)
    sp.rename(columns= {'obswave'  : 'wave'}, inplace=True)
    sp['flam'] = sp['flam'].astype(np.float64)   #force to be a float, not a str
    sp['flam_u'] = sp['flam_u'].astype(np.float64)   #force to be a float, not a str
    sp['wave'] = sp['wave'].astype(np.float64)   #force to be a float, not a str
    sp['fnu']     = jrr.spec.flam2fnu(sp.wave, sp.flam)          # convert fnu to flambda
    sp['fnu_u']   = jrr.spec.flam2fnu(sp.wave, sp.flam_u)
    if getclean:
        sp.badmask = sp.badmask.astype(bool)
        sp = sp[~sp['badmask']]
    sp['fnu_autocont'] = pd.Series(np.ones_like(sp.wave)*np.nan)  # Will fill this with automatic continuum fit
    return sp   # Returns the spectrum as a Pandas data frame, the spectral resoln as a float, and its uncertainty

#-------------Function to get the list of lines to fit--------------------------
def getlist(listname, zz_dic, zz_err_dic):
    '''
    NOTE: Use labframe.shortlinelist to include most lines except a few and
    Use labframe.shortlinelist_com to exclude most lines  and fit a few
    '''
    LL = pd.read_table(listname, delim_whitespace=True, comment="#") # Load different, shorter linelist to fit lines
    LL = LL.sort('restwave')
    LL.restwave = LL['restwave'].astype(np.float64)
    line_full = pd.DataFrame(columns=['zz','zz_err','type'])
    for kk in range(0, len(LL)):
        t = LL.iloc[kk].type
        row = np.array([zz_dic[t], zz_err_dic[t], t])
        line_full.loc[len(line_full)] = row
    line_full.zz = line_full.zz.astype(np.float64)
    line_full.zz_err = line_full.zz_err.astype(np.float64)
    line_full.insert(0,'label',LL.LineID)
    line_full.insert(1,'wave', LL.restwave*(1.+line_full.zz))
    line_full.wave = line_full.wave.astype(np.float64)
    return line_full #pandas dataframe

#------------Function to calculate MAD error spectrum for entire spectrum and add column to dataframe-----------------
def calc_mad(sp, resoln, nn):
    start = np.min(sp.wave)
    bin_edges = [start]
    while start < np.max(sp.wave):
        end = start + sp[sp.wave >= start].wave.values[0]*gs2f/resoln
        bin_edges.append(end)
        start = end
    bin_med, bin_edges, binnumber = stats.binned_statistic(sp.wave, sp.flam, statistic='median', bins=np.array(bin_edges))
    bin_mad, bin_centers = [], []
    for ii in range(0,len(bin_med)/nn):
        bin_mad.append(stat.mad(bin_med[nn*ii:nn*(ii+1)]))
        bin_centers.append(bin_edges[1+(nn/2)*ii] - (bin_edges[1+(nn/2)*ii]-bin_edges[(nn/2)*ii])/2.)
    bin_mad = jrr.spec.flam2fnu(np.array(bin_centers), np.array(bin_mad))
    #func = interp1d(bin_centers, bin_mad, kind='cubic')
    func = extrapolate(bin_centers, bin_mad, k=3) #order 3, cubic
    plt.plot(bin_centers, bin_mad, c='red')
    plt.draw()
    sp['mad'] = pd.Series(func(sp.wave))
    print np.shape(sp.mad), np.shape(sp.wave), np.shape(func(sp.wave)) #
    plt.plot(sp.wave, func(sp.wave), c='blue')
    plt.xlim(np.min(bin_centers),np.max(bin_centers))
    plt.ylim(np.array([-0.1,1.])*1e-28)
    plt.show(block=False)

#------------Function to calculate Schneider EW and errors at every point in spectrum and add columns to dataframe-----------------
def calc_schneider_EW(sp, resoln, plotit = False):
    EW, sig, sig_int, signorm_int = [],[],[], []
    w = sp.wave.values
    f = (sp.flam/sp.flam_autocont).values
    #--normalised error spectrum for EW limit----
    unorm = (sp.flam_u/sp.flam_autocont).values #normalised flux error
    func_u = interp1d(w, unorm, kind='linear')
    uinorm = func_u(w) #interpolated normalised flux error
    #---------------------------
    disp = np.concatenate(([np.diff(w)[0]],np.diff(w))) #disperion array
    n = len(w)
    lim = 3.
    N = 2.
    for ii in range(len(w)):
        b = w[ii]
        c = w[ii]*gf2s/resoln       
        j0 = int(np.round(N*c/disp[ii]))
        a = 1./np.sum([exp(-((disp[ii]*(j0-j))**2)/(2*c**2)) for j in range(2*j0+1)])
        P = [a*exp(-((disp[ii]*(j0-j))**2)/(2*c**2)) for j in range(2*j0+1)]
        j1 = max(1,j0-ii)
        j2 = min(2*j0, j0+(n-1)-ii)
        #For reference of following equations, please see 1st and 3rd equation of section 6.2 of Schneider et al. 1993.
        #The 2 quantities on the left side of those equations correspond to 'EW' and 'signorm_int' respectively which subsequently become 'W_interp' and 'W_u_interp'
        EW.append(disp[ii]*np.sum(P[j]*(f[ii+j-j0]-1.)for j in range(j1, j2+1))/np.sum(P[j]**2 for j in range(j1, j2+1)))
        signorm_int.append(disp[ii]*np.sqrt(np.sum(P[j]**2*uinorm[ii+j-j0]**2for j in range(j1, j2+1)))/np.sum(P[j]**2 for j in range(j1, j2+1)))
        #sig.append(disp[ii]*np.sqrt(np.sum(P[j]**2*unorm[ii+j-j0]**2for j in range(j1, j2+1)))/np.sum(P[j]**2 for j in range(j1, j2+1)))
    func_ew = interp1d(w, EW, kind='linear')
    W = func_ew(w) #interpolating EW
    #sp['W_interp'] = pd.Series(W) #'W_interp' is the result of interpolation of the weighted rolling average of the EW (based on the SSF chosen)
    #sp['W_u_interp'] = pd.Series(signorm_int) #'W_u_interp' is 1 sigma error in EW derived by weighted rolling average of interpolated flux error.
    sp['W_interp'] = W
    sp['W_u_interp'] = signorm_int
    #-----plot if you want to see-------------------
    if plotit:
        fig = plt.figure(figsize=(14,6))
        plt.plot(w, f, c='blue', label='normalised flux')
        plt.plot(w, unorm, c='gray', label='flux error')
        plt.plot(w, W, c='red', label='interpolated EW')
        plt.plot(w, np.zeros(len(w)), c='g', linestyle='--', label='zero level')
        plt.plot(w, np.multiply(lim,signorm_int), c='g', label=str(int(lim))+'sig EW err')
        plt.plot(w, -np.multiply(lim,signorm_int), c='g')
        plt.xlim(4200,4600)
        plt.ylim(-1,3)
        plt.xlabel('Observed wavelength (A)')
        plt.legend()
        plt.show(block=False)
    return 0 #just so the function returns something

#-------------Fucntion for updating popt, pcov when some parameters are fixed----------------------------
#-------------so that eventually update_dataframe() gets popt, pcov of usual shape----------------------------
def update_p(popt, pcov, pos, popt_insert, n, pcov_insert=0.):
    for yy in range(0, n):
        #print popt, pos+4*yy #Debugging
        popt = np.insert(popt,pos+4*yy,popt_insert)
        if pcov_insert == 0.:
            pcov = np.insert(np.insert(pcov,pos+4*yy,pcov_insert, axis=0),pos+4*yy,pcov_insert, axis=1)
        else:
            pcov = np.insert(np.insert(pcov,pos+4*yy,pcov[0], axis=0),pos,np.insert(pcov[:,0],pos+4*yy,pcov_insert), axis=1)
    return popt, pcov

#-------------Fucntion for deciding in a line is detected or not----------------------------
def isdetect(EW_signi, f_signi, f_SNR, EW_thresh=None, f_thresh=None, f_SNR_thresh=None):
    if EW_thresh is not None:
        if EW_signi >= EW_thresh:
            if f_SNR_thresh is not None:                    #will check if f_SNR is above threshold if threshold is present
                if f_SNR >= f_SNR_thresh: return True
                else: return False
            else: return True
        else: return False
    elif f_thresh is not None:
        if f_signi >= f_thresh:
            if f_SNR_thresh is not None:                    #will check if f_SNR is above threshold if threshold is present
                if f_SNR >= f_SNR_thresh: return True
                else: return False
            else: return True
        else: return False
    else:
        return isdetect(EW_signi, f_signi, f_SNR, EW_thresh=3.) #if neither f thresholds specified, re-calling isdetect() with a default EW_thresh=3

#-------------Fucntion for fitting any number of Gaussians----------------------------
def fit(sptemp, l, resoln, dresoln, fix_cen = False, fix_cont=False):
    global v_maxwidth, line_type_dic
    types=[]
    for v in l.type.values:
        types.append(line_type_dic[v])
    #fitting 3 parameters, keeping center fixed
    if fix_cen:
        p_init, lbound, ubound = [sptemp.flam.values[0]],[-np.inf],[np.inf]
        for xx in range(0, len(l)):
            fl = sptemp[sptemp['wave']>=l.wave.values[xx]].flam.values[0]-cont
            p_init = np.append(p_init, [np.abs(fl)*types[xx], l.wave.values[xx]*2.*gf2s/resoln])
            lbound = np.append(lbound,[-np.inf if float(types[xx] < 0) else 0., l.wave.values[xx]*1.*gf2s/(resoln-3.*dresoln)])
            ubound = np.append(ubound,[np.inf if float(types[xx] > 0) else 0., l.wave.values[xx]*v_maxwidth*gf2s/3e5])
        popt, pcov = curve_fit(lambda x, *p: s.fixcen_gaus(x, l.wave.values, *p),sptemp['wave'],sptemp['flam'],p0=p_init, sigma = sptemp['flam_u'])
        for yy in range(0, len(l)-1):
            popt, pcov = update_p(popt, pcov, 3, popt[0], 1, pcov_insert=pcov[0][0])
            popt, pcov = update_p(popt, pcov, 2, l.wave.values, 1)
        popt, pcov = update_p(popt, pcov, 2, l.wave.values[-1], 1)
        
    #fitting 3 parameters, keeping continuum fixed
    elif fix_cont:
        p_init, lbound, ubound =[],[],[]
        cont = 1.
        for xx in range(0, len(l)):
            if 'Ly-alpha' in l.label.values[xx]:
                zz_allow = (2000./3e5) + 3.*l.zz_err.values[xx] # increasing allowance of z (by 2000 km/s) if the line to be fit is Ly-alpha
            elif 'MgII' in l.label.values[xx]:
                zz_allow = (300./3e5) + 3.*l.zz_err.values[xx] # increasing allowance of z (by 300 km/s) if the line to be fit is MgII2797 lines, due to wind
            else:
                 zz_allow = 3.*l.zz_err.values[xx]
            fl = sptemp[sptemp['wave']>=l.wave.values[xx]].flam.values[0]-cont
            p_init = np.append(p_init, [np.abs(fl)*types[xx], l.wave.values[xx], l.wave.values[xx]*np.mean([1.*gf2s/(resoln-3.*dresoln), v_maxwidth*gf2s/3e5])])#2.*gf2s/resoln])
            lbound = np.append(lbound,[-np.inf if float(types[xx] < 0) else 0.,l.wave.values[xx]*(1.-zz_allow/(1.+l.zz.values[xx])),l.wave.values[xx]*1.*gf2s/(resoln+3.*dresoln)])
            ubound = np.append(ubound,[np.inf if float(types[xx] > 0) else 0.,l.wave.values[xx]*(1.+zz_allow/(1.+l.zz.values[xx])),l.wave.values[xx]*v_maxwidth*gf2s/3e5])
        popt, pcov = curve_fit(lambda x, *p: s.fixcont_gaus(x, cont, len(l), *p),sptemp['wave'],sptemp['flam'], p0 = p_init, sigma = sptemp['flam_u'], absolute_sigma = True, bounds = (lbound, ubound))
        popt, pcov = update_p(popt, pcov, 0, cont, len(l))
    
    #fitting all 4 parameters, nothing fixed
    else:
        p_init, lbound, ubound = [ly[0]],[-np.inf],[np.inf]
        for xx in range(0, len(l)):
            zz_allow = (2000./3e5) + 3.*l.zz_err.values[xx] if 'Ly-alpha' in l.label.values[xx] else 3.*l.zz_err.values[xx] # increasing allowance of z (by 2000 km/s) if the line to be fit is Ly-alpha
            fl = sptemp[sptemp['wave']>=l.wave.values[xx]].flam.values[0]-cont
            p_init = np.append(p_init, [np.abs(fl)*types[xx], l.wave.values[xx], l.wave.values[xx]*2.*gf2s/resoln])
            lbound = np.append(lbound,[-np.inf if float(types[xx] < 0) else 0., l.wave.values[xx]*(1.-zz_allow/(1.+l.zz.values[xx])),l.wave.values[xx]*1.*gf2s/(resoln-3.*dresoln)])
            ubound = np.append(ubound,[np.inf if float(types[xx] > 0) else 0.,l.wave.values[xx]*(1.+zz_allow/(1.+l.zz.values[xx])),l.wave.values[xx]*v_maxwidth*gf2s/3e5])
        popt, pcov = curve_fit(lambda x, *p: s.gaus(x, len(l), *p),sptemp['wave'],sptemp['flam'],p0= p_init, sigma = sptemp['flam_u'])
        popt, pcov = update_p(popt, pcov, 4, popt[0], len(l)-1, pcov_insert=pcov[0][0])
    return popt, pcov
    
#-------------Function for updating the linelist data frame by adding each line------------
def update_dataframe(sp, label, l, df, resoln, dresoln, popt=None, pcov=None, fit_successful=True, EW_thresh=None, f_thresh=None, f_SNR_thresh=None):
    global line_type_dic
    #------calculating EW using simple summation------
    wv, f, f_c, f_u = s.cutspec(sp.wave, sp.flam*sp.flam_autocont, sp.flam_autocont, sp.flam_u*sp.flam_autocont, l.wave*(1.-2.*gs2f/(resoln-dresoln)),  l.wave*(1.+2.*gs2f/(resoln-dresoln))) # +/- 2 sigma ->FWHM
    disp = [j-i for i, j in zip(wv[:-1], wv[1:])]
    EWr_sum, EWr_sum_u = np.array(jrr.spec.calc_EW(f[:-1], f_u[:-1], f_c[:-1], 0., disp, l.zz)) #*1.05 #aperture correction sort of
    #-------3sigma limit from Schneider et al. 1993 prescription------------
    sign = (line_type_dic[l.type]) #to take care of emission/absorption
    EWr_3sig_lim = -1.*sign*3.*(sp.loc[sp.wave >= l.wave].W_u_interp.values[0])/(1+l.zz) #dividing by (1+z) as the dataframe has observed frame EW limits
    fl_3sig_lim = -1.*EWr_3sig_lim*sp.loc[sp.wave >= l.wave].flam_autocont.values[0]
    #--------------------------------------------------
    if fit_successful:
        cont = sp.loc[sp.wave >= popt[2]].flam_autocont.values[0] #continuum value at the line centre
        EWr_fit = np.sqrt(2*np.pi)*(-1.)*popt[1]*popt[3]/(popt[0]*(1.+l.zz)) #convention: -ve EW is EMISSION
        EWr_fit_u = np.sqrt(2*np.pi*(pcov[1][1]*(popt[3]/popt[0])**2 + pcov[3][3]*(popt[1]/popt[0])**2 + pcov[0][0]*(popt[1]*popt[3]/popt[0]**2)**2))/(1.+l.zz)
        zz = popt[2]*(1.+l.zz)/l.wave - 1.
        zz_u = np.sqrt(pcov[2][2])*(1.+l.zz)/l.wave
        f_line = np.sqrt(2*np.pi)*popt[1]*popt[3]*cont #total flux = integral of guassian fit
        f_line_u = np.sqrt(2*np.pi*(pcov[1][1]*popt[3]**2 + pcov[3][3]*popt[1]**2))*cont #multiplied with cont at that point in wavelength to get units back in ergs/s/cm^2
        EW_signi = 3.*EWr_fit/EWr_3sig_lim # computing significance of detection in EW
        f_signi = 3.*f_line/fl_3sig_lim # computing significance of detection in flux
        f_SNR = f_line/f_line_u
        detection = isdetect(EW_signi, f_signi, f_SNR, EW_thresh=EW_thresh, f_thresh=f_thresh, f_SNR_thresh=f_SNR_thresh)
        #this is where all the parameters of a measured line is put into the final dataframe (called line_table in EW_fitter.py)
        #please note that variables EWr_3sig_lim and fl_3sig_lim here are referred to as Ewr_Suplim and f_Suplim respectively, in EW_fitter.py
        row = np.array([label, l.label, ("%.4f" % l.wave), ("%.4f" % float(l.wave/(1+l.zz))), l.type, \
        ("%.4f" % EWr_fit), ("%.4f" % EWr_fit_u), ("%.4f" % EWr_sum), ("%.4f" % EWr_sum_u), ("%.4e" % f_line),\
        ("%.4e" % f_line_u), ("%.4f" % EWr_3sig_lim), ("%.3f" % EW_signi), ("%.4e" % fl_3sig_lim), ("%.3f" % f_signi), ("%.4f" % popt[0]), ("%.4f" % popt[1]), \
        ("%.4f" % popt[2]), ("%.4f" % np.sqrt(pcov[2][2])), ("%.4f" % popt[3]), ("%.5f" % zz), ("%.5f" % zz_u)])
    else:
        detection = False
        row = np.array([label, l.label, ("%.4f" % l.wave), ("%.4f" % float(l.wave/(1+l.zz))), l.type, np.nan, np.nan, \
        ("%.4f" % EWr_sum), ("%.4f" % EWr_sum_u), np.nan, np.nan, ("%.4f" % EWr_3sig_lim),\
        ("%.4e" % fl_3sig_lim), np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
    df.loc[len(df)] = row
    return detection
#-------Function to plot the gaussians-------------
def plot_gaus(sptemp, popt, cen, label, zz, tot_fl, detection=True, silent = True, plotfnu=False):
    gauss_curve_lam = np.multiply(s.gaus1(sptemp.wave,*popt),sptemp.flam_autocont)
    gauss_curve_nu = jrr.spec.flam2fnu(sptemp.wave, gauss_curve_lam)
    if detection:
        if not plotfnu: plt.plot(sptemp.wave, gauss_curve_lam, color='red', linewidth=1, linestyle = '-')
        else: plt.plot(sptemp.wave, gauss_curve_nu, color='red', linewidth=1, linestyle = '-')
        plt.axvline(popt[2], c='r', lw=0.5)
        plt.text(popt[2]+1, plt.gca().get_ylim()[-1]*0.9, label, color='r', rotation=90)
        if not silent: print 'Detected', label
    else:
        if not plotfnu: plt.plot(sptemp.wave, gauss_curve_lam, color='k', linewidth=1, linestyle = '--')
        else: plt.plot(sptemp.wave, gauss_curve_nu, color='k', linewidth=1, linestyle = '--')
        plt.axvline(popt[2], c='k', lw=1)
        plt.text(popt[2]+1, plt.gca().get_ylim()[-1]*0.9, label, color='k', rotation=90)
        if not silent: print 'NOT detected', label
    plt.axvline(cen, c='blue', lw=0.5)
    if not plotfnu: tot_fl += gauss_curve_lam
    else: tot_fl += gauss_curve_nu
    return tot_fl

#-------Functions to correct for extinction for rcs0327-E ONLY-------------
def kappa(w, i):
    if i==0:
        k = 0
    elif i==1:
        k = 2.659*(-2.156+1.509/w-0.198/(w**2)+0.011/(w**3))+4.05
    elif i==2:
        k = 2.659*(-1.857 + 1.040/w) + 4.05
    return k
#------------------Function to calculate extinction and de-redden fluxes-------------
def extinct(wave, flux, flux_u, E, E_u, inAngstrom=True):
    if inAngstrom: wave/=1e4 #to convert to micron
    wbreaks = [0.12, 0.63, 2.2]
    flux_redcor,flux_redcor_u=[],[]
    for i,wb in enumerate(wbreaks):
        w = wave.iloc[np.where(wave<wb)[0]]
        f = flux.iloc[np.where(wave<wb)[0]]
        f_u = flux_u.iloc[np.where(wave<wb)[0]]
        k = kappa(w,i)
        fredcor = np.multiply(f,10**(0.4*k*E))
        fredcor_u = np.multiply(10**(0.4*k*E),np.sqrt(f_u**2 + (f*0.4*k*np.log(10)*E_u)**2)) #error propagation
        flux_redcor.extend(fredcor)
        flux_redcor_u.extend(fredcor_u)
        ind = np.where(wave<wb)[0][-1] if len(np.where(wave<wb)[0]) > 0 else -1
        wave = wave.iloc[ind+1:]
        flux = flux.iloc[ind+1:]
        flux_u = flux_u.iloc[ind+1:]
    return flux_redcor, flux_redcor_u

#-------Function to calculate one sigma error in flux at certain wavelength--------
#------------------NOT USED ANYMORE-----------------------------
def calc_1sig_err(sptemp, cen, resoln):
    dpix = sptemp.wave.values[2] - sptemp.wave.values[1]
    dlambda = 5.*cen/resoln
    sptemp = sptemp[sptemp['wave'].between(cen-dlambda, cen+dlambda)]
    try:
        err_wt_mean = np.power(np.sum(1/((sptemp['flam_u']*sptemp['flam_autocont']*dpix)**2.+(0.)**2)),-0.5)
        wt_mean = np.sum((sptemp['flam']*sptemp['flam_autocont']*dpix)/((sptemp['flam_u']*sptemp['flam_autocont']*dpix)**2.))/(err_wt_mean**-2.)
    except ZeroDivisionError:
        return 999
    return wt_mean, err_wt_mean
    
#-------Function to check if good (3-sigma) detection------
#------------------NOT USED ANYMORE-----------------------------
def check_3sig_det(sptemp, l, popt, resoln, args=None):
    f_line = np.abs(np.sqrt(2*np.pi)*popt[1]*popt[3])*sptemp[sptemp.wave >= popt[2]].flam_autocont.values[0]
    wt_mean, err_wt_mean = calc_1sig_err(sptemp, l.wave, resoln)
    if f_line >= 3.*np.abs(err_wt_mean):
        return True, wt_mean, err_wt_mean
    else:
        if not args.silent:
            print 'Non detection of', l.label #
        return False, wt_mean, err_wt_mean
        
#-------Function to calculate med_bin_flux and mad_bin_flux to get upper limit on detection------
#------------------NOT USED ANYMORE-----------------------------
def calc_detec_lim(sp_orig, line, resoln, nbin, args=None):
    dlambda = 2.*gs2f/resoln
    leftlim = line.wave.values[0]*(1.-5./resoln)*(1.-dlambda)
    rightlim = line.wave.values[-1]*(1.+5./resoln)*(1.+dlambda)
    l_arr = np.concatenate((np.linspace(leftlim*(1. - (nbin-1)*dlambda), leftlim, nbin),np.linspace(rightlim, rightlim*(1. + (nbin-1)*dlambda), nbin)))
    fluxes = []
    for l in l_arr:
        sptemp = sp_orig[sp_orig['wave'].between(l*(1.-dlambda),  l*(1.+dlambda))] #this is sp_orig, hence NOT continuum normalised
        if args.showbin:
            try:
                plt.axvline(np.min(sptemp.wave), c='red', lw=0.5, linestyle='--')
            except:
                pass
            try:
                plt.axvline(np.max(sptemp.wave), c='red', lw=0.5, linestyle='--')
            except:
                pass
        try:
            dpix = sptemp.wave.values[2] - sptemp.wave.values[1]
        except IndexError:
            dpix = 0.3 #from other parts of the spectra
        fluxes.append(np.sum((sptemp['flam']-sptemp['flam_autocont'])*dpix))
    return np.median(fluxes), stat.mad(fluxes)

#-------------Fucntion for fitting multiple lines----------------------------
def fit_some_EWs(line, sp, resoln, label, df, dresoln, sp_orig, args=None) :
    # This is what Ayan needs to fill in, from his previous code.
    # Should work on sp.wave, sp.flam, sp.flam_u, sp.flam_autocont
    global v_maxwidth, line_type_dic
    line_type_dic = {'EMISSION':1., 'FINESTR':1., 'PHOTOSPHERE': -1., 'ISM':-1., 'WIND':-1.}
    if args.fcen is not None:
        fix_cen = int(args.fcen)
    else:
        fix_cen = 0
    if args.fcon is not None:
        fix_con = int(args.fcon)
    else:
        fix_cont = 1
    if args.vmax is not None:
        v_maxwidth = float(args.vmax)
    else:
        v_maxwidth = 300. #in km/s, to set the maximum FWHM that can be fit as a line
    if args.nbin is not None:
        nbin = int(args.nbin)
    else:
        nbin = 5
    #-----------
    kk, c = 1, 0
    ndlambda_left, ndlambda_right = [5.]*2 #how many delta-lambda wide will the window (for line fitting) be on either side of the central wavelength, default 5
    try:
        c = 1
        first, last = [line.wave.values[0]]*2
        if 'Ly-alpha' in line.label.values[0]: #treating Ly-alpha specially: widening the wavelength window
            ndlambda_left, ndlambda_right = 3., 12.
    except IndexError:
        pass
    while kk <= len(line):
        center1 = last
        if kk == len(line):
            center2 = 1e10 #insanely high number, required to plot last line
        else:
            center2 = line.wave.values[kk]
        if center2*(1. - 5./resoln) > center1*(1. + 5./resoln):
            sp2 = sp[sp['wave'].between(first*(1.-ndlambda_left/resoln), last*(1.+ndlambda_right/resoln))]
            sp2.flam = sp2.flam/sp2.flam_autocont #continuum normalising by autocont
            sp2.flam_u = sp2.flam_u/sp2.flam_autocont #continuum normalising by autocont

            if not args.showbin:
                plt.axvline(np.min(sp2.wave), c='blue', lw=0.5, linestyle='--')
                plt.axvline(np.max(sp2.wave), c='blue', lw=0.5, linestyle='--')
            
            if not args.silent:
                print 'Trying to fit', line.label.values[kk-c:kk], 'line/s at once. Total', c
            #med_bin_flux, mad_bin_flux = calc_detec_lim(sp_orig, line[kk-c:kk], resoln, nbin, args=args) #NOT REQUIRED anymore
            try:
                popt, pcov = fit(sp2, line[kk-c:kk], resoln, dresoln, fix_cont=fix_cont, fix_cen=fix_cen)
                tot_fl = np.zeros(len(sp2))
                for xx in range(0,c):
                    ind = line.index.values[(kk-1) - c + 1 + xx]
                    #det_3sig, wt_mn, er_wt_mn = check_3sig_det(sp2, line.loc[ind], popt[4*xx:4*(xx+1)], resoln, args=args) # check if 3 sigma detection; NOT REQUIRED anymore
                    detection = update_dataframe(sp2, label, line.loc[ind], df, resoln, dresoln, popt= popt[4*xx:4*(xx+1)], pcov= pcov[4*xx:4*(xx+1),4*xx:4*(xx+1)], fit_successful=True)
                    tot_fl = plot_gaus(sp2, popt[4*xx:4*(xx+1)], line.loc[ind].wave, line.loc[ind].label, line.loc[ind].zz, tot_fl, detection=detection, silent = args.silent, plotfnu = args.plotfnu)
                if c > 1:
                        if not args.plotfnu: plt.plot(sp2.wave, np.subtract(tot_fl,(c-1.)*np.multiply(popt[0],sp2.flam_autocont)), color='green', linewidth=2)
                        else: plt.plot(sp2.wave, np.subtract(tot_fl,(c-1.)*jrr.spec.flam2fnu(sp2.wave, np.multiply(popt[0],sp2.flam_autocont))), color='green', linewidth=2)
                if not args.silent:
                    print 'done above fitting'
            
            except (RuntimeError, ValueError, IndexError) as e:
                for xx in range(0,c):
                    ind = line.index.values[(kk-1) - c + 1 + xx]
                    plt.axvline(line.loc[ind].wave, c='k', lw=1)
                    #wt_mn, er_wt_mn = calc_1sig_err(sp2, line.loc[ind].wave, resoln) #NOT REQUIRED anymore
                    try:
                        dummy = update_dataframe(sp2, label, line.loc[ind], df, resoln, dresoln, fit_successful=False)
                    except:
                        pass
                if not args.silent:
                    print 'Could not fit these', c, 'lines.'
                    print 'Error in ayan.mage.fit_some_EWs:', e #
            
            first, last = [center2]*2
            if kk < len(line) and 'Ly-alpha' in line.label.values[kk]: #treating Ly-alpha specially: widening the wavelength window
                ndlambda_left, ndlambda_right = 3., 12.
            c = 1
        else:
            last = center2
            if kk < len(line) and 'Ly-alpha' in line.label.values[kk]: #treating Ly-alpha specially: widening the wavelength window
                ndlambda_right = 12.
            c += 1
        kk += 1
              
    return df #df is a pandas data frame that has the line properties

#-------------End of functions----------------------------
