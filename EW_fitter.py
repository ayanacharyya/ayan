''' Equivalent width and flux fitter.  Given a spectrum, a continuum, a linelist, and a redshift,
fit a bunch of emission or absorption lines.
Started july 2016, Jane Rigby and Ayan Acharyya
'''
import mage as m
import sys
sys.path.append('/Users/acharyya/Dropbox/MagE_atlas/Tools')
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
import argparse as ap
#-----------Main function starts------------------
parser = ap.ArgumentParser(description="Mage spectra fitting tool")
parser.add_argument("--shortlabel")
parser.add_argument("--fcen")
parser.add_argument("--fcon")
parser.add_argument("--dx")
parser.add_argument("--only")
parser.add_argument("--vmax")
parser.add_argument("--frame")
parser.add_argument("--nbin")
parser.add_argument('--keepprev', dest='keepprev', action='store_true')
parser.set_defaults(keepprev=False)
parser.add_argument('--silent', dest='silent', action='store_true')
parser.set_defaults(silent=False)
parser.add_argument('--mask', dest='mask', action='store_true')
parser.set_defaults(mask=False)
parser.add_argument('--check', dest='check', action='store_true')
parser.set_defaults(check=False)
parser.add_argument('--allspec', dest='allspec', action='store_true')
parser.set_defaults(allspec=False)
parser.add_argument('--savepdf', dest='savepdf', action='store_true')
parser.set_defaults(savepdf=False)
parser.add_argument('--hide', dest='hide', action='store_true')
parser.set_defaults(hide=False)
parser.add_argument('--showbin', dest='showbin', action='store_true')
parser.set_defaults(showbin=False)
parser.add_argument('--fullmad', dest='fullmad', action='store_true')
parser.set_defaults(fullmad=False)
parser.add_argument('--showerr', dest='showerr', action='store_true')
parser.set_defaults(showerr=False)
args, leftovers = parser.parse_known_args()
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
if args.shortlabel is not None:
    labels = [args.shortlabel]
else:
    labels = ['rcs0327-E']
if args.allspec:
    labels = [
    'rcs0327-B',\
    'rcs0327-E',\
    #'rcs0327-Ehires',\
    #'rcs0327-Elores',\
    'rcs0327-G',\
    'rcs0327-U',\
    #'rcs0327-BDEFim1',\
    #'rcs0327-counterarc',\
    'S0004-0103',\
    #'S0004-0103alongslit',\
    #'S0004-0103otherPA',\
    'S0033+0242',\
    'S0108+0624',\
    'S0900+2234',\
    'S0957+0509',\
    'S1050+0017',\
    'Horseshoe',\
    'S1226+2152',\
    #'S1226+2152hires',\
    #'S1226+2152lores',\
    'S1429+1202',\
    'S1458-0023',\
    'S1527+0652',\
    #'S1527+0652-fnt',\
    'S2111-0114',\
    'Cosmic~Eye',\
    #'S2243-0935',\
    ]
if not args.keepprev:
    plt.close('all')
#-------------------------------------------------------------------------
(specs) = jrr.mage.getlist_labels(mage_mode, labels)
(spec_path, line_path) = jrr.mage.getpath(mage_mode)
line_table = pd.DataFrame(columns=['label', 'line_lab', 'obs_wav', 'rest_wave', 'type','EWr_fit','EWr_fit_u', 'EWr_sum', \
'EWr_sum_u', 'f_line','f_line_u', \
#'wt_mn_flux', 'onesig_err_wt_mn_flux', \
'med_bin_flux', 'mad_bin_flux', 'significance', 'EW_3sig_lim_Schneider', 'fit_cont','fit_fl','fit_cen', 'fit_cen_u', \
'fit_sig','zz','zz_u'])

for ii in range(0, len(specs)) :                  #nfnu_stack[ii] will be ii spectrum
    shortlabel     = specs['short_label'][ii]
    print 'Spectrum', (ii+1), 'of', len(specs),':', shortlabel #Debugging
    filename  = specs['filename'][ii]
    zz_sys = specs['z_syst'][ii] # from the new z_syst column in spectra_filename file
    zz_dic = {'EMISSION':specs['z_neb'][ii], 'FINESTR':specs['z_neb'][ii], 'PHOTOSPHERE': specs['z_stars'][ii] if specs['fl_st'][ii]==0 else specs['z_neb'][ii], 'ISM':specs['z_ISM'][ii], 'WIND':specs['z_ISM'][ii]}
    zz_err_dic = {'EMISSION':specs['sig_neb'][ii] if specs['fl_neb'][ii]==0 else specs['sig_ISM'][ii], 'FINESTR':specs['sig_neb'][ii] if specs['fl_neb'][ii]==0 else specs['sig_ISM'][ii], 'PHOTOSPHERE': specs['sig_st'][ii] if specs['fl_st'][ii]==0 else specs['sig_neb'][ii], 'ISM':specs['sig_ISM'][ii], 'WIND':specs['sig_ISM'][ii]}    
    (sp_orig, resoln, dresoln)  = jrr.mage.open_spectrum(filename, zz_sys, mage_mode)
    #-------masking sky lines-----------------
    if args.mask:
        sp_orig['badmask']  = False
        m.flag_skylines(sp_orig)
        sp_orig = sp_orig[~sp_orig.badmask].copy(deep=True) #masking for skylines
    #---------------------------------------
    m.fit_autocont(sp_orig, zz_sys, line_path,filename)
    sp_orig = sp_orig[~sp_orig['badmask']]
    if args.fullmad:
        sp_orig = m.calc_mad(sp_orig, resoln, 5)
        continue
    sp_orig = m.calc_schneider_EW(sp_orig, resoln, plotit=args.showerr) # calculating the EW limits at every point following Schneider et al. 1993
    #makelist(linelist)
    line_full = m.getlist(zz_dic, zz_err_dic)
    #------------Preparing to plot----------------------------------------
    xstart = max(np.min(line_full.wave) - 50.,np.min(sp_orig.wave))
    xlast = min(np.max(line_full.wave) + 50.,np.max(sp_orig.wave))
    if frame is None:
        n_arr = np.arange(int(np.ceil((xlast-xstart)/dx))).tolist()
    else:
        n_arr = [int(frame)] #Use this to display a single frame
    name = '/Users/acharyya/Desktop/mage_plot/'+shortlabel+'_CNOSi_emission_fit'
    if args.savepdf:
        pdf = PdfPages(name+'.pdf')
    #---------pre check in which frames lines are available if display_only_success = 1---------
    #---------------------------just a visualisation thing-----------------------------------
    if display_only_success:
        for jj in n_arr:
            xmin = xstart + jj*dx
            xmax = min(xmin + dx, xlast)
            sp = sp_orig[sp_orig['wave'].between(xmin,xmax)]
            try:
                line = line_full[line_full['wave'].between(xmin*(1.+5./resoln), xmax*(1.-5./resoln))]
            except IndexError:
                continue
            if not len(line) > 0 or not line['wave'].between(np.min(sp.wave),np.max(sp.wave)).all():
                n_arr[jj] = np.ma.masked
        n_arr = np.ma.compressed(n_arr)
    #------------------------------------------------------------
    n = len(n_arr) #number of frames that would be displayed
    fig = plt.figure(figsize=(18+8/(n+1),(18 if n > 2 else n*6)))
    #fig = plt.figure(figsize=(16+8/(n+1),(8 if n > 2 else n*3)))
    plt.title(shortlabel + "  z=" + str(zz_sys)+'. Vertical lines legend: Blue=initial guess of center,'+\
    ' Red=fitted center, Green=no detection(< 3sigma), Black=unable to fit gaussian', y=1.02)
    for fc, jj in enumerate(n_arr):
        xmin = xstart + jj*dx
        xmax = min(xmin + dx, xlast)
        ax1 = fig.add_subplot(n,1,fc+1)
        sp = sp_orig[sp_orig['wave'].between(xmin,xmax)]
        try:
            line = line_full[line_full['wave'].between(xmin*(1.+5./resoln), xmax*(1.-5./resoln))]
        except IndexError:
            continue
        #------------Plot the results------------
        plt.step(sp.wave, sp.fnu, color='b')
        plt.step(sp.wave, sp.fnu_u, color='gray')
        plt.step(sp.wave, sp.fnu_cont, color='y')
        plt.plot(sp.wave, sp.fnu_autocont, color='k')
        plt.ylim(0, 1.2E-28)
        plt.xlim(xmin, xmax)
        plt.text(xmin+dx*0.05, ax1.get_ylim()[1]*0.8, 'Frame '+str(int(jj)))
        if not args.fullmad:
            line_table = m.fit_some_EWs(line, sp, resoln, shortlabel, line_table, dresoln, sp_orig, args=args) #calling line fitter
    
        ax2 = ax1.twiny()
        ax2.set_xlim(ax1.get_xlim())       
        ax2.set_xticklabels(np.round(np.divide(ax1.get_xticks(),(1.+zz_sys)),decimals=0))        
        labels2 = [item.get_text() for item in ax2.get_xticklabels()]
        ax2.set_xticks(np.concatenate([ax2.get_xticks(), line.wave*(1.+zz_sys)/(1.+line.zz)]))
        ax2.set_xticklabels(np.concatenate([labels2,np.array(line.label.values).tolist()]), rotation = 45, ha='left', fontsize='small')
        fig.subplots_adjust(hspace=0.7, top=0.95, bottom=0.05)
    fig.tight_layout()
    if not args.hide:
        plt.show(block=False)
    if args.savepdf:
        pdf.savefig(fig)
        pdf.close()
    #fig.savefig(name+'.png')
#------------changing data types------------------------------
line_table.obs_wav = line_table.obs_wav.astype(np.float64)
line_table.rest_wave = line_table.rest_wave.astype(np.float64)
line_table.EWr_fit = line_table.EWr_fit.astype(np.float64)
line_table.EWr_fit_u = line_table.EWr_fit_u.astype(np.float64)
line_table.EWr_sum = line_table.EWr_sum.astype(np.float64)
line_table.EWr_sum_u = line_table.EWr_sum_u.astype(np.float64)
line_table.f_line = line_table.f_line.astype(np.float64)
line_table.f_line_u = line_table.f_line_u.astype(np.float64)
#line_table.wt_mn_flux = line_table.wt_mn_flux.astype(np.float64)
#line_table.onesig_err_wt_mn_flux = line_table.onesig_err_wt_mn_flux.astype(np.float64)
line_table.med_bin_flux = line_table.med_bin_flux.astype(np.float64)
line_table.mad_bin_flux = line_table.mad_bin_flux.astype(np.float64)
line_table.significance = line_table.significance.astype(np.float64)
line_table.EW_3sig_lim_Schneider = line_table.EW_3sig_lim_Schneider.astype(np.float64)
line_table.fit_cont = line_table.fit_cont.astype(np.float64)
line_table.fit_fl = line_table.fit_fl.astype(np.float64)
line_table.fit_cen = line_table.fit_cen.astype(np.float64)
line_table.fit_cen_u = line_table.fit_cen_u.astype(np.float64)
line_table.fit_sig = line_table.fit_sig.astype(np.float64)
line_table.zz = line_table.zz.astype(np.float64)
line_table.zz_u = line_table.zz_u.astype(np.float64)
#------------------------------------------------------------
if not args.hide:
    print line_table
else:
    line_table['f_SNR']=np.abs(line_table['f_line'])/line_table['f_line_u']
    line_table['EW_significance']=3.*line_table['EWr_fit']/line_table['EW_3sig_lim_Schneider']
    print line_table[['line_lab','f_SNR','significance','EWr_fit','EWr_fit_u','EW_3sig_lim_Schneider','EW_significance']]

fout = '/Users/acharyya/Dropbox/mage_atlas/Tools/Contrib/fitted_line_list_new.txt'
head = 'This file contains the measurements of lines in the MagE sample. Generated by EW_fitter.py.\n\
Columns are:\n\
label: shortlabel of the galaxy/knot\n\
line_lab: label of the line the code was asked to fit\n\
obs_wav: observed frame wavelength of the line (A)\n\
rest_wave: rest frame wavelength of the line (A)\n\
type: is it emission or ism etc.\n\
EWr_fit: eqv width as calculated from the Gaussian fit to the line (A)\n\
EWr__fit_u: error in above qty. (A)\n\
EWr_sum: eqv width as calculated by summing the flux (A)\n\
EWr_sum_u: error in above qty. (A)\n\
f_line: flux i.e. area under Gaussian fit (erg/s/cm^2)\n\
f_line_u: error in above qty. (erg/s/cm^2)\n\
wt_mn_flux: weighted mean flux at the center of the line (erg/s/cm^2)\n\
onesig_err_wt_mn_flux: 1 sigma uncertainty in weighted mean flux at the center of the line (erg/s/cm^2)\n\
fit_cont: continuum, as from the fit (continuum normalised fit)\n\
fit_fl: amplitude, as from the fit (continuum normalised fit)\n\
fit_cen: center, as from the fit (continuum normalised fit)\n\
fit_cen_u: error in above qty. (A)\n\
fit_sig: sigma, as from the fit (A)\n\
zz: Corrected redshift of this line, from the fitted center\n\
zz_u: error in above qty.\n\
NaN means either the code was unable to fit the line OR the fit was below 3 sigma flux error at that point\n\
'
np.savetxt(fout, [], header=head, comments='#')
line_table.to_csv(fout, sep='\t',mode ='a', index=None)
print 'Table saved to', fout
#----------------Sanity check: comparing 2 differently computed EWs------------------
if args.check:
    err_sum, es, n = 0., 0., 0
    for p in range(0,len(line_table)):
        EWr_fit = float(line_table.iloc[p].EWr_fit)
        EWr_sum = float(line_table.iloc[p].EWr_sum)
        EWr_sum_u = float(line_table.iloc[p].EWr_sum_u)
        #print line_table.iloc[p].line_lab, EWr_fit, EWr_sum, EWr_sum_u #Debugging
        if np.abs(EWr_sum) > EWr_sum_u and EWr_fit > 0.:
            err_sum += EWr_fit - EWr_sum
            es += (EWr_fit - EWr_sum)**2.
            n += 1
    if n > 0:
        print err_sum/n, np.sqrt(es/n)
    else:
        print 'No lines detected.'
#------------------------------------------End of main function------------------------------------------------