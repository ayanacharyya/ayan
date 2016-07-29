import sys
sys.path.append('/Users/acharyya/Dropbox/MagE_atlas/Tools')
sys.path.append('/Users/acharyya/Dropbox/MagE_atlas/Tools/Contrib')
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

# added a comment for testing
foobar_variable = False

#-------Function for making new linelist files. DO NOT DELETE--------------------------
def makelist(linelist):    
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
     'N II 1084', 'N II 1085', \
     'N I 1134a', 'N I 1134b', 'N I 1134c', \
     'N III 1183', 'N III 1184', 'N IV 1169', 'N III 1324', 'N IV 1718',\
     'N IV] 1486', 'N II] 2140',\
     ]
    (LL, zz_redundant) = jrr.mage.get_linelist(linelist)
    line_full = pd.DataFrame(columns=['label','wave','type','source'])
    for label in target_line_labels_to_fit:
        try:
            t = LL[(LL['lab1']+' '+LL['lab2']).eq(label) & LL['type'].ne('INTERVE')].type.values[0]
        except IndexError:
            print label, 'not found'
            continue
        row = np.array([''+label.replace(' ', ''), LL[(LL['lab1']+' '+LL['lab2']).eq(label) & LL['type'].ne('INTERVE')].restwav.values[0], t, 'Leitherer'])
        line_full.loc[len(line_full)] = row
    fout = 'labframe.shortlinelist'
    line.to_csv(fout, sep='\t',mode ='a', index=None)

    fn = open('/Users/acharyya/Mappings/lab/targetlines.txt','r')
    fout2 = open(fout,'a')
    for lin in fn.readlines():
        if len(lin.split())>1 and lin.split()[0][0] != '#':
            fout2.write(''+lin.split()[1]+'   '+lin.split()[0]+'   EMISSION'+'   MAPPINGS\n')
    fout2.close()
    fn.close()
    LL = pd.read_table(fout, delim_whitespace=True, comment='#') # Load different, shorter linelist to fit lines
    LL = LL.sort('restwave')
    np.savetxt(fout, np.transpose([LL.LineID, LL.restwave, LL.type, LL.source]), "%s   %s   %s   %s", \
    header=head, comments='')
    subprocess.call(['python /Users/acharyya/Desktop/mage_plot/comment_file.py '+'/Users/acharyya/Dropbox/MagE_atlas/Tools/Contrib/'+fout],shell=True)
#-----------Function to flag skylines, by JRR----------------------
def flag_skylines(sp) :
    # Mask skylines [O I] 5577\AA\ and [O I]~6300\AA,
    skyline = (5577., 6300.)
    skywidth = 7.0  # flag spectrum +- skywidth of the skyline, skywidth different from JRR
    for thisline in skyline:
        sp.badmask.loc[sp['wave'].between(thisline-skywidth, thisline+skywidth)] = True 

#-------------Function to fit autocont using jrr.mage.auto_fit.cont------------------
def fit_autocont(sp_orig, zz_sys, line_path, filename):
    linelist = jrr.mage.get_linelist_name(filename, line_path)   # convenience function
    (LL, zz_redundant) = jrr.mage.get_linelist(linelist)  # Load the linelist to fit auto-cont    
    jrr.mage.auto_fit_cont(sp_orig, LL, zz_sys)  # Automatically fit continuum.  results written to sp.fnu_autocont, sp.flam_autocont.

#-------------Function to get the list of lines to fit--------------------------
def getlist(zz_dic, zz_err_dic):
    '''
    NOTE: Use labframe.shortlinelist to include most lines except a few and
    Use labframe.shortlinelist_com to exclude most lines  and fit a few
    '''
    LL = pd.read_table('labframe.shortlinelist_com', delim_whitespace=True, comment="#") # Load different, shorter linelist to fit lines
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

#------------Function to calculate error spectrum-----------------
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
    return sp

#------------Function to calculate error spectrum-----------------
def calc_schneider_EW(sp, resoln, plotit = False):
    EW, sig, sig_int = [],[],[]
    w = sp.wave.values
    f = (sp.flam/sp.flam_autocont).values
    u = (sp.flam_u/sp.flam_autocont).values
    func_u = interp1d(w, u, kind='linear')
    ui = func_u(w)
    disp = np.concatenate(([np.diff(w)[0]],np.diff(w)))
    n = len(w)
    lim = 3.
    N = 2.
    for ii in range(len(w)):
        b = w[ii]
        c = w[ii]*gf2s/resoln       
        j0 = int(np.round(N*c/disp[ii]))
        #a = 1./np.sum([exp(-((w[ii-j0+j]-w[ii])**2)/(2*c**2)) for j in range(2*j0+1)])
        #P = [a*exp(-((w[ii-j0+j]-w[ii])**2)/(2*c**2)) for j in range(2*j0+1)]
        a = 1./np.sum([exp(-((disp[ii]*(j0-j))**2)/(2*c**2)) for j in range(2*j0+1)])
        P = [a*exp(-((disp[ii]*(j0-j))**2)/(2*c**2)) for j in range(2*j0+1)]
        j1 = max(1,j0-ii)
        j2 = min(2*j0, j0+(n-1)-ii)
        EW.append(disp[ii]*np.sum(P[j]*(f[ii+j-j0]-1.)for j in range(j1, j2+1))/np.sum(P[j]**2 for j in range(j1, j2+1)))
        sig.append(disp[ii]*np.sqrt(np.sum(P[j]**2*u[ii+j-j0]**2for j in range(j1, j2+1)))/np.sum(P[j]**2 for j in range(j1, j2+1)))
        sig_int.append(disp[ii]*np.sqrt(np.sum(P[j]**2*ui[ii+j-j0]**2for j in range(j1, j2+1)))/np.sum(P[j]**2 for j in range(j1, j2+1)))
    func_ew = interp1d(w, EW, kind='linear')
    W = func_ew(w)
    sp['W_interp'] = pd.Series(W)
    sp['W_u_interp'] = pd.Series(sig_int)
    #-----plot if you want to see-------------------
    if plotit:
        plt.plot(w, f, c='blue')
        plt.plot(w, u, c='gray')
        plt.plot(w, W, c='red')
        plt.plot(w, np.zeros(len(w)), c='g', linestyle='--')
        plt.plot(w, np.multiply(lim,sig_int), c='g')
        plt.plot(w, -np.multiply(lim,sig_int), c='g')
        plt.xlim(4200,4600)
        plt.ylim(-1,3)
        plt.show(block=False)
    #---------------------------------------------
    return sp
    
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
        #print sptemp[~np.isfinite(sptemp['flam'])] #
        p_init, lbound, ubound =[],[],[]
        cont = 1.
        for xx in range(0, len(l)):
            fl = sptemp[sptemp['wave']>=l.wave.values[xx]].flam.values[0]-cont
            p_init = np.append(p_init, [np.abs(fl)*types[xx], l.wave.values[xx], l.wave.values[xx]*2.*gf2s/resoln])
            lbound = np.append(lbound,[-np.inf if float(types[xx] < 0) else 0.,l.wave.values[xx]*(1.-3.*l.zz_err.values[xx]/(1.+l.zz.values[xx])),l.wave.values[xx]*1.*gf2s/(resoln-3.*dresoln)])
            ubound = np.append(ubound,[np.inf if float(types[xx] > 0) else 0.,l.wave.values[xx]*(1.+3.*l.zz_err.values[xx]/(1.+l.zz.values[xx])),l.wave.values[xx]*v_maxwidth*gf2s/3e5])
        popt, pcov = curve_fit(lambda x, *p: s.fixcont_gaus(x, cont, len(l), *p),sptemp['wave'],sptemp['flam'], p0 = p_init, sigma = sptemp['flam_u'], absolute_sigma = True, bounds = (lbound, ubound))
        #print pcov #Debugging
        popt, pcov = update_p(popt, pcov, 0, cont, len(l))
    #fitting all 4 parameters, nothing fixed
    else:
        p_init, lbound, ubound = [ly[0]],[-np.inf],[np.inf]
        for xx in range(0, len(l)):
            fl = sptemp[sptemp['wave']>=l.wave.values[xx]].flam.values[0]-cont
            p_init = np.append(p_init, [np.abs(fl)*types[xx], l.wave.values[xx], l.wave.values[xx]*2.*gf2s/resoln])
            lbound = np.append(lbound,[-np.inf if float(types[xx] < 0) else 0., l.wave.values[xx]*(1.-3.*l.z_err.values[xx]/(1.+l.zz.values[xx])),l.wave.values[xx]*1.*gf2s/(resoln-3.*dresoln)])
            ubound = np.append(ubound,[np.inf if float(types[xx] > 0) else 0.,l.wave.values[xx]*(1.+3.*l.z_err.values[xx]/(1.+l.zz.values[xx])),l.wave.values[xx]*v_maxwidth*gf2s/3e5])
        popt, pcov = curve_fit(lambda x, *p: s.gaus(x, len(l), *p),sptemp['wave'],sptemp['flam'],p0= p_init, sigma = sptemp['flam_u'])
        popt, pcov = update_p(popt, pcov, 4, popt[0], len(l)-1, pcov_insert=pcov[0][0])
    return popt, pcov
    
#-------------Function for updating the data frame for each line------------
def update_dataframe(sp, label, l, med_bin_flux, mad_bin_flux, df, resoln, dresoln, popt=None, pcov=None, detection=True):
    global line_type_dic
    #------calculating EW using simple summation------
    wv, f, f_c, f_u = s.cutspec(sp.wave, sp.flam*sp.flam_autocont, sp.flam_autocont, sp.flam_u*sp.flam_autocont, l.wave*(1.-2.*gs2f/(resoln-dresoln)),  l.wave*(1.+2.*gs2f/(resoln+dresoln))) # +/- 2 sigma ->FWHM
    disp = [j-i for i, j in zip(wv[:-1], wv[1:])]
    EWr_sum, EWr_sum_u = np.array(jrr.spec.calc_EW(f[:-1], f_u[:-1], f_c[:-1], 0., disp, l.zz))*1.05 #aperture correction sort of
    #-------3sigma limit from Schneider et al. 1993 prescription------------
    sign = (line_type_dic[l.type]) #to take care of emission/absorption
    EW_3sig_lim = -1.*sign*3.*(sp.loc[sp.wave >= l.wave].W_u_interp.values[0])/(1+l.zz) #dividing by (1+z) as the dataframe has observed frame EW limits
    #--------------------------------------------------
    if detection:
        cont = sp.loc[sp.wave >= popt[2]].flam_autocont.values[0]
        EWr_fit = np.sqrt(2*np.pi)*(-1.)*popt[1]*popt[3]/(popt[0]*(1.+l.zz)) #convention: -ve EW is EMISSION
        EWr_fit_u = np.sqrt(2*np.pi*(pcov[1][1]*(popt[3]/popt[0])**2 + pcov[3][3]*(popt[1]/popt[0])**2 + pcov[0][0]*(popt[1]*popt[3]/popt[0]**2)**2))/(1.+l.zz)
        zz = popt[2]*(1.+l.zz)/l.wave - 1.
        zz_u = np.sqrt(pcov[2][2])*(1.+l.zz)/l.wave
        f_line = np.sqrt(2*np.pi)*popt[1]*popt[3]*cont #total flux = integral of guassian fit
        f_line_u = np.sqrt(2*np.pi*(pcov[1][1]*popt[3]**2 + pcov[3][3]*popt[1]**2))*cont #multiplied with cont at that point in wavelength to get units back in ergs/s/cm^2
        signi = sign*(f_line - med_bin_flux)/mad_bin_flux
        row = np.array([label, l.label, ("%.4f" % l.wave), ("%.4f" % float(l.wave/(1+l.zz))), l.type, \
        ("%.4f" % EWr_fit), ("%.4f" % EWr_fit_u), ("%.4f" % EWr_sum), ("%.4f" % EWr_sum_u), ("%.4e" % f_line),\
        ("%.4e" % f_line_u), \
        #("%.4e" % wt_mn), ("%.4e" % er_wt_mn), \
        ("%.4e" % med_bin_flux),("%.4e" % mad_bin_flux), ("%.4e" % signi), ("%.4f" % EW_3sig_lim),\
        ("%.4f" % popt[0]), ("%.4f" % popt[1]), ("%.4f" % popt[2]), \
        ("%.4f" % np.sqrt(pcov[2][2])), ("%.4f" % popt[3]), ("%.4f" % zz), ("%.4f" % zz_u)])
    else:
        row = np.array([label, l.label, ("%.4f" % l.wave), ("%.4f" % float(l.wave/(1+l.zz))), l.type, np.nan, np.nan, \
        ("%.4f" % EWr_sum), ("%.4f" % EWr_sum_u), np.nan, np.nan, \
        #("%.4e" % wt_mn), ("%.4e" % er_wt_mn), \
        ("%.4e" % med_bin_flux),("%.4e" % mad_bin_flux), np.nan, ("%.4f" % EW_3sig_lim),\
        np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
    df.loc[len(df)] = row
    return df
    
#-------Function to plot the gaussians-------------
def plot_gaus(sptemp, popt, cen, tot_nu, detection=True):
    gauss_curve_lam = np.multiply(s.gaus1(sptemp.wave,*popt),sptemp.flam_autocont)
    gauss_curve_nu = jrr.spec.flam2fnu(sptemp.wave, gauss_curve_lam)
    if detection:
        plt.plot(sptemp.wave, gauss_curve_nu, color='red', linewidth=1, linestyle = '--')
        plt.axvline(popt[2], c='r', lw=0.5)
    else:
        plt.plot(sptemp.wave, gauss_curve_nu, color='k', linewidth=1, linestyle = '--')
        plt.axvline(popt[2], c='g', lw=1)
    plt.axvline(cen, c='blue', lw=0.5)
    tot_nu += gauss_curve_nu
    return tot_nu

#-------Function to calculate one sigma error in flux at certain wavelength--------
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
def check_3sig_det(sptemp, l, popt, resoln, args=None):
    f_line = np.abs(np.sqrt(2*np.pi)*popt[1]*popt[3])*sptemp[sptemp.wave >= popt[2]].flam_autocont.values[0]
    wt_mean, err_wt_mean = calc_1sig_err(sptemp, l.wave, resoln)
    if f_line >= 3.*np.abs(err_wt_mean):
        return True, wt_mean, err_wt_mean
    else:
        if not args.silent:
            print 'Non detection of', l.label #
        return False, wt_mean, err_wt_mean
        
#-------Function to calculate upper limit on detection------
def calc_detec_lim(sp_orig, line, resoln, nbin, args=None):
    dlambda = 0.5*gs2f/resoln
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
'''
def calc_detec_lim(sp_orig, line):
    dlambda = 0.5*gs2f/resoln
    l_arr = np.linspace(line.wave*(1. - nbin*dlambda), line.wave*(1. + nbin*dlambda), 2*nbin + 1)
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
        dpix = sptemp.wave.values[2] - sptemp.wave.values[1]
        fluxes.append(np.sum((sptemp['flam']-sptemp['flam_autocont'])*dpix))
    return np.median(fluxes), stat.mad(fluxes)
'''
#-------------Fucntion for fitting multiple lines----------------------------
def fit_some_EWs(line, sp, resoln, label, df, dresoln, sp_orig, args=None) :
    # This is what Ayan needs to fill in, from his previous code.
    # Should work on sp.wave, sp.flam, sp.flam_u, sp.flam_autocont
    global v_maxwidth, line_type_dic
    line_type_dic = {'EMISSION':1., 'FINESTR':1., 'PHOTOSPHERE': -1., 'ISM':-1., 'WIND':-1.}
    if args.fcen is not None:
        fix_cen = args.fcen
    else:
        fix_cen = 0
    if args.fcon is not None:
        fix_con = args.fcon
    else:
        fix_cont = 1
    if args.dx is not None:
        dx = args.dx
    else:
        dx = 310.
    if args.only is not None:
       display_only_success = args.only
    else:
        display_only_success = 1
    if args.vmax is not None:
        v_maxwidth = args.vmax
    else:
        v_maxwidth = 300 #in km/s, to set the maximum FWHM that can be fit as a line
    if args.frame is not None:
        frame = args.frame
    else:
        frame = None
    if args.nbin is not None:
        nbin = args.nbin
    else:
        nbin = 5
    #-----------
    kk, c = 1, 0
    try:
        c = 1
        first, last = line.wave.values[0], line.wave.values[0]
    except IndexError:
        pass
    while kk <= len(line):
        center1 = last
        if kk == len(line):
            center2 = 1e10 #insanely high number, required to plot last line
        else:
            center2 = line.wave.values[kk]
        if center2*(1. - 5./resoln) > center1*(1. + 5./resoln):
            sp2 = sp[sp['wave'].between(first*(1.-5./resoln), last*(1.+5./resoln))]
            sp2.flam = sp2.flam/sp2.flam_autocont #continuum normalising by autocont
            sp2.flam_u = sp2.flam_u/sp2.flam_autocont #continuum normalising by autocont
            if not args.showbin:
                plt.axvline(np.min(sp2.wave), c='blue', lw=0.5, linestyle='--')
                plt.axvline(np.max(sp2.wave), c='blue', lw=0.5, linestyle='--')
            
            if not args.silent:
                print 'Trying to fit', line.label.values[kk-c:kk], 'line/s at once. Total', c
            med_bin_flux, mad_bin_flux = calc_detec_lim(sp_orig, line[kk-c:kk], resoln, nbin, args=args)
            try:
                popt, pcov = fit(sp2, line[kk-c:kk], resoln, dresoln, fix_cont=fix_cont, fix_cen=fix_cen)
                tot_nu = np.zeros(len(sp2))
                for xx in range(0,c):
                    ind = line.index.values[(kk-1) - c + 1 + xx]
                    #det_3sig, wt_mn, er_wt_mn = check_3sig_det(sp2, line.loc[ind], popt[4*xx:4*(xx+1)], resoln, args=args) # check if 3 sigma detection
                    det_3sig = True
                    df = update_dataframe(sp2, label, line.loc[ind], med_bin_flux, mad_bin_flux, df, resoln, dresoln, popt= popt[4*xx:4*(xx+1)], pcov= pcov[4*xx:4*(xx+1),4*xx:4*(xx+1)], detection=det_3sig)
                    tot_nu = plot_gaus(sp2, popt[4*xx:4*(xx+1)], line.loc[ind].wave, tot_nu, detection=det_3sig)
                if c > 1:
                        plt.plot(sp2.wave, np.subtract(tot_nu,(c-1.)*jrr.spec.flam2fnu(sp2.wave, np.multiply(popt[0],sp2.flam_autocont))), color='green', linewidth=2)
                if not args.silent:
                    print 'done above fitting'
            except (RuntimeError, ValueError):
                for xx in range(0,c):
                    ind = line.index.values[(kk-1) - c + 1 + xx]
                    plt.axvline(line.loc[ind].wave, c='k', lw=1)
                    #wt_mn, er_wt_mn = calc_1sig_err(sp2, line.loc[ind].wave, resoln)
                    df = update_dataframe(sp2, label, line.loc[ind], med_bin_flux, mad_bin_flux, df, resoln, dresoln, detection=False)
                if not args.silent:
                    print 'Could not fit these', c, 'lines.'                
            first, last = center2, center2
            c = 1
        else:
            last = center2
            c += 1
        kk += 1
              
    return df #df is a pandas data frame that has the line properties

#-------------End of functions----------------------------
