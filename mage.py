'''
Collection of python routines by Ayan acharyya applicable on mage sample, mainly to be used by the code EW_fitter.py
to fit gaussian profiles to spectral lines.
Started July 2016
'''
import os

HOME = os.getenv('HOME') + '/'
import sys

sys.path.append(HOME + 'Dropbox/MagE_atlas/Tools')
sys.path.append(HOME + 'Dropbox/MagE_atlas/Tools/Contrib')
import jrr
import splot_util as s
import numpy as np
import pandas as pd

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1000)
from astropy.io import ascii
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
from scipy import asarray as ar, exp


# -------Function for making new linelist files. DO NOT DELETE even if unused--------------------------
def makelist(linelist):
    # in the following list, keep (comment out rest) all the lines you want to be extracted from Rigby's .linelist files
    # into our new labframe.shortlinelist files
    target_line_labels_to_fit = [
        'C III 1247', \
        'Si III 1294', 'Si III 1296', 'C III 1296', 'Si III 1298', 'O I 1302', 'Si II 1304', \
        'Si II 1309', 'O I 1304', 'O I 1306', \
        'C III] 1906', 'C III] 1908', \
        'C II 1334', 'C II* 1335', 'C II 1323', 'N III 1324', 'O IV 1341', \
        # 'O IV 1343', \
        'Si III 1417', \
        'S V 1501', \
        'C III 1620', \
        'Fe IV 1717', \
        'Fe III 1954', 'Fe III 1958', \
        'C II] 2326', \
        'He II 1640', 'O III] 1666', \
        'Si II 1533', \
        'C III 2297', \
        '[O III] 2321', \
        '[O III] 2331', \
        'O III] 1666', \
        'O III] 1660', \
        'N II 1084', 'N II 1085', \
        'N I 1134a', 'N I 1134b', 'N I 1134c', \
        'N III 1183', 'N III 1184', 'N IV 1169', 'N III 1324', 'N IV 1718', \
        'N IV] 1486', 'N II] 2140', \
        'Mn II 2576', 'Fe II 2586', 'Mn II 2594', 'Fe II 2599', 'Fe II 2600', \
        'Mn II 2606', 'Fe II 2607', 'Fe II 2612', 'Fe II 2614', 'Fe II 2618', 'Fe II 2621', 'Fe II 2622', 'Fe II 2626',
        'Fe II 2629', 'Fe II 2631', 'Fe II 2632', \
        'Mg II 2796', 'Mg II 2803', 'Mg I 2852', \
        'Ti II 3073', 'Ti II 3230', 'Ti II 3239', 'Ti II 3242', 'Ti II 3384', \
        'Ca II 3934', 'Ca II 3969', 'Ca I 4227', \
        'Na I 5891', 'Na I 5897', \
        'Li I 6709' \
        ]
    (LL, zz_redundant) = jrr.mage.get_linelist(linelist)
    line_full = pd.DataFrame(columns=['LineID', 'restwave', 'type', 'source'])
    for label in target_line_labels_to_fit:
        try:
            t = LL[(LL['lab1'] + ' ' + LL['lab2']).eq(label) & LL['type'].ne('INTERVE')].type.values[0]
        except IndexError:
            print label, 'not found'
            continue
        row = np.array(['' + label.replace(' ', ''),
                        LL[(LL['lab1'] + ' ' + LL['lab2']).eq(label) & LL['type'].ne('INTERVE')].restwav.values[0], t,
                        'Leitherer'])
        line_full.loc[len(line_full)] = row
    fout = 'labframe.shortlinelist'
    line_full.to_csv(fout, sep='\t', mode='a', index=None)

    fn = open(HOME + 'Mappings/lab/targetlines.txt', 'r')
    fout2 = open(fout, 'a')
    for lin in fn.readlines():
        if len(lin.split()) > 1 and lin.split()[0][0] != '#':
            fout2.write('' + lin.split()[1] + '   ' + lin.split()[0] + '   EMISSION' + '   MAPPINGS\n')
    fout2.close()
    fn.close()
    LL = pd.read_table(fout, delim_whitespace=True, comment='#')  # Load different, shorter linelist to fit lines
    LL = LL.sort_values('restwave')
    head = '#New customised linelist mostly only for emission lines\n\
#Columns are:\n\
LineID  restwave    type    source\n'

    np.savetxt(fout, np.transpose([LL.LineID, LL.restwave, LL.type, LL.source]), "%s   %s   %s   %s", \
               header=head, comments='')
    print 'Created new linelist', fout
    # subprocess.call(['python /Users/acharyya/Desktop/mage_plot/comment_file.py '+'/Users/acharyya/Dropbox/MagE_atlas/Tools/Contrib/'+fout],shell=True)


# -------Function for making new intervenning-linelist files. DO NOT DELETE even if unused--------------------------
def make_interven_list(linefile, zz_interv=0.01):
    ll = pd.read_csv(linefile, sep='\s+', header=None, names=['restwave', 'lab1', 'lab2', 'zz', 'dummy', 'type'])
    ll.insert(0, 'LineID', (ll.lab1 + ll.lab2.astype(str)))
    ll.drop('lab1', axis=1, inplace=True)
    ll.drop('lab2', axis=1, inplace=True)
    ll.drop('dummy', axis=1, inplace=True)
    ll.insert(4, 'source', ['JRR_interven.lst'] * len(ll))
    ll['zz'].replace('xxxx', str(zz_interv),
                     inplace=True)  # replacing redshifts of unknown intervenning absorptions with zz_interv
    # zz_interv chosen such that FeII2600 leads to an intervenning line between CII1906,8 for rcs0327-E (z=1.7032)
    fout = 'labframe.shortlinelist_interven'
    head = '#New customised linelist mostly only for intervenning lines\n\
#Columns are:\n'
    ll.to_csv(fout, sep='\t', mode='w', header=head, index=None)
    print 'Created new linelist', fout


# -----------Function to flag skylines in a different way than by JRR, if required----------------------
def flag_skylines(sp):
    # Mask skylines [O I] 5577\AA\ and [O I]~6300\AA,
    skyline = np.array([5577., 6300.])
    prev_skywidth = 17.0  # used by jrr.mage.flag_skylines
    skywidth = skyline * 250. / 3e5  # flag spectrum +- skywidth of the skyline, skywidth different from JRR
    # masking vel = 250 km/s on either side of skylines
    for pp in range(len(skyline)):
        sp.badmask.loc[sp['wave'].between(skyline[pp] - prev_skywidth,
                                          skyline[pp] + prev_skywidth)] = False  # undo jrr.mage masking
        sp.badmask.loc[
            sp['wave'].between(skyline[pp] - skywidth[pp], skyline[pp] + skywidth[pp])] = True  # redo new masking


# -------------Function to fit autocont using jrr.mage.auto_fit.cont------------------
def fit_autocont(sp_orig, zz_sys, line_path, filename, boxcar=1001):
    if 'stack' in filename.lower():
        linelist = line_path + 'stacked.linelist'  # to provide line list for jrr.mage.auto_fit_cont to mask out regions of the spectra
    elif 'new-format' in filename.lower():
        linelist = line_path + 'stacked.linelist'
    elif 'esi' in filename.lower():
        linelist = line_path + 'stacked.linelist'
    else:
        linelist = jrr.mage.get_linelist_name(filename, line_path)  # convenience function
    (LL, zz_redundant) = jrr.mage.get_linelist(linelist)  # Load the linelist to fit auto-cont
    LL.zz = zz_sys # set the systemic redshift column of the linelist (which is zero by default) to the systemic redshift of the concerned galaxy
    jrr.spec.fit_autocont(sp_orig, LL, zz_sys, colv2mask='vmask', boxcar=boxcar, colmask='contmask')  # Automatically fit continuum.  results written to sp.fnu_autocont, sp.flam_autocont.


# -------------Function to read in specra file of format (obswave, fnu, fnu_u, restwave), based on jrr.mage.open_spectrum------------------
def open_esi_spectrum(infile, getclean=True, flamcol='flam', flamucol='flam_u', obswavecol='obswave'):
    '''Reads a reduced ESI spectrum
      Inputs:   filename to read in
      Outputs:  the object spectrum, in both flam and fnu (why not?) all in Pandas data frame
      Pandas keys:  wave, fnu, fnu_u, flam, flam_u
      call:  (Pandas_spectrum_dataframe, spectral_resolution) = ayan.mage.open_spectrum(infile)
    '''
    sp = pd.read_table(infile, delim_whitespace=True, comment="#", header=0)  # , names=names)
    sp.rename(columns={obswavecol: 'wave'}, inplace=True)
    sp.rename(columns={flamcol: 'flam'}, inplace=True)
    sp.rename(columns={flamucol: 'flam_u'}, inplace=True)
    sp['flam'] = sp['flam'].astype(np.float64)  # force to be a float, not a str
    sp['flam_u'] = sp['flam_u'].astype(np.float64)  # force to be a float, not a str
    sp['wave'] = sp['wave'].astype(np.float64)  # force to be a float, not a str
    sp['fnu'] = jrr.spec.flam2fnu(sp.wave, sp.flam)  # convert fnu to flambda
    sp['fnu_u'] = jrr.spec.flam2fnu(sp.wave, sp.flam_u)
    if getclean:
        sp.badmask = sp.badmask.astype(bool)
        sp = sp[~sp['badmask']]
    sp['fnu_autocont'] = pd.Series(np.ones_like(sp.wave) * np.nan)  # Will fill this with automatic continuum fit
    return sp  # Returns the spectrum as a Pandas data frame, the spectral resoln as a float, and its uncertainty


# -------------Function to get the list of lines to fit--------------------------
def getlist(listname, zz_dic, zz_err_dic):
    '''
    NOTE: Use labframe.shortlinelist to include most lines except a few and
    Use labframe.shortlinelist_com to exclude most lines  and fit a few
    '''
    LL = pd.read_table(listname, delim_whitespace=True, comment="#")  # Load different, shorter linelist to fit lines
    '''
    Reads in table of the format:
    LineID  restwave    type    source
    '''
    LL = LL.sort_values('restwave')
    LL.restwave = LL['restwave'].astype(np.float64)
    line_full = pd.DataFrame(columns=['zz', 'zz_err', 'type'])
    for kk in range(0, len(LL)):
        t = LL.iloc[kk].type
        row = np.array([zz_dic[t], zz_err_dic[t], t])
        line_full.loc[len(line_full)] = row
    line_full.zz = line_full.zz.astype(np.float64)
    line_full.zz_err = line_full.zz_err.astype(np.float64)
    line_full.insert(0, 'label', LL.LineID)
    line_full.insert(1, 'wave', LL.restwave * (1. + line_full.zz))
    line_full.wave = line_full.wave.astype(np.float64)
    line_full['vmask'] = 500.0  # default window to mask the line, in par with jrr.mage.get_linelist()
    return line_full  # pandas dataframe


# -------------Function to get the list of lines to fit--------------------------
def get_interven_list(listname, zz_err=0.0004):
    LL = pd.read_table(listname, delim_whitespace=True, comment="#")  # Load different, shorter linelist to fit lines
    '''
    Reads in table of the format:
    LineID  restwave    zz      type    source
    '''
    LL = LL.sort_values('restwave')
    LL.rename(columns={'LineID': 'label'}, inplace=True)
    LL.restwave *= (1 + LL.zz)
    LL.rename(columns={'restwave': 'wave'}, inplace=True)
    LL.insert(3, 'zz_err', pd.Series(np.ones(len(LL)) * zz_err))
    LL.drop('source', axis=1, inplace=True)
    return LL  # pandas dataframe


# ------------Function to calculate MAD error spectrum for entire spectrum and add column to dataframe-----------------
def calc_mad(sp, resoln, nn):
    start = np.min(sp.wave)
    bin_edges = [start]
    while start < np.max(sp.wave):
        end = start + sp[sp.wave >= start].wave.values[0] * gs2f / resoln
        bin_edges.append(end)
        start = end
    bin_med, bin_edges, binnumber = stats.binned_statistic(sp.wave, sp.flam, statistic='median',
                                                           bins=np.array(bin_edges))
    bin_mad, bin_centers = [], []
    for ii in range(0, len(bin_med) / nn):
        bin_mad.append(stat.mad(bin_med[nn * ii:nn * (ii + 1)]))
        bin_centers.append(
            bin_edges[1 + (nn / 2) * ii] - (bin_edges[1 + (nn / 2) * ii] - bin_edges[(nn / 2) * ii]) / 2.)
    bin_mad = jrr.spec.flam2fnu(np.array(bin_centers), np.array(bin_mad))
    # func = interp1d(bin_centers, bin_mad, kind='cubic')
    func = extrapolate(bin_centers, bin_mad, k=3)  # order 3, cubic
    plt.plot(bin_centers, bin_mad, c='red')
    plt.draw()
    sp['mad'] = pd.Series(func(sp.wave))
    print np.shape(sp.mad), np.shape(sp.wave), np.shape(func(sp.wave))  #
    plt.plot(sp.wave, func(sp.wave), c='blue')
    plt.xlim(np.min(bin_centers), np.max(bin_centers))
    plt.ylim(np.array([-0.1, 1.]) * 1e-28)
    plt.show(block=False)


# ------------Function to calculate Schneider EW and errors at every point in spectrum and add columns to dataframe-----------------
def calc_schneider_EW(sp, resoln, plotit=False):
    EW, sig, sig_int, signorm_int = [], [], [], []
    w = sp.wave.values
    f = (sp.flam / sp.flam_autocont).values
    # --normalised error spectrum for EW limit----
    unorm = (sp.flam_u / sp.flam_autocont).values  # normalised flux error
    func_u = interp1d(w, unorm, kind='linear')
    uinorm = func_u(w)  # interpolated normalised flux error
    # ---------------------------
    disp = np.concatenate(([np.diff(w)[0]], np.diff(w)))  # disperion array
    n = len(w)
    lim = 3.
    N = 2.
    for ii in range(len(w)):
        b = w[ii]
        c = w[ii] * gf2s / resoln
        j0 = int(np.round(N * c / disp[ii]))
        a = 1. / np.sum([exp(-((disp[ii] * (j0 - j)) ** 2) / (2 * c ** 2)) for j in range(2 * j0 + 1)])
        P = [a * exp(-((disp[ii] * (j0 - j)) ** 2) / (2 * c ** 2)) for j in range(2 * j0 + 1)]
        j1 = max(1, j0 - ii)
        j2 = min(2 * j0, j0 + (n - 1) - ii)
        # For reference of following equations, please see 1st and 3rd equation of section 6.2 of Schneider et al. 1993.
        # The 2 quantities on the left side of those equations correspond to 'EW' and 'signorm_int' respectively which subsequently become 'W_interp' and 'W_u_interp'
        EW.append(disp[ii] * np.sum(P[j] * (f[ii + j - j0] - 1.) for j in range(j1, j2 + 1)) / np.sum(
            P[j] ** 2 for j in range(j1, j2 + 1)))
        signorm_int.append(
            disp[ii] * np.sqrt(np.sum(P[j] ** 2 * uinorm[ii + j - j0] ** 2 for j in range(j1, j2 + 1))) / np.sum(
                P[j] ** 2 for j in range(j1, j2 + 1)))
        # sig.append(disp[ii]*np.sqrt(np.sum(P[j]**2*unorm[ii+j-j0]**2for j in range(j1, j2+1)))/np.sum(P[j]**2 for j in range(j1, j2+1)))
    func_ew = interp1d(w, EW, kind='linear')
    W = func_ew(w)  # interpolating EW
    # sp['W_interp'] = pd.Series(W) #'W_interp' is the result of interpolation of the weighted rolling average of the EW (based on the SSF chosen)
    # sp['W_u_interp'] = pd.Series(signorm_int) #'W_u_interp' is 1 sigma error in EW derived by weighted rolling average of interpolated flux error.
    sp['W_interp'] = W
    sp['W_u_interp'] = signorm_int
    # -----plot if you want to see-------------------
    if plotit:
        fig = plt.figure(figsize=(14, 6))
        plt.plot(w, f, c='blue', label='normalised flux')
        plt.plot(w, unorm, c='gray', label='flux error')
        plt.plot(w, W, c='red', label='interpolated EW')
        plt.plot(w, np.zeros(len(w)), c='g', linestyle='--', label='zero level')
        plt.plot(w, np.multiply(lim, signorm_int), c='g', label=str(int(lim)) + 'sig EW err')
        plt.plot(w, -np.multiply(lim, signorm_int), c='g')
        plt.xlim(4200, 4600)
        plt.ylim(-1, 3)
        plt.xlabel('Observed wavelength (A)')
        plt.legend()
        plt.show(block=False)
    return 0  # just so the function returns something


# -------------Fucntion for updating popt, pcov when some parameters are fixed----------------------------
# -------------so that eventually update_dataframe() gets popt, pcov of usual shape----------------------------
def update_p(popt, pcov, pos, n, popt_insert, pcov_insert, extend=False):
    '''
    This function is to generalise the shape of the output parameter array.
    For e.g. if we were fitting all 4 (continuum, height, width, center) gaussian parameters, for N lines
    our popt array would be a 1D array of length 4xN and pcov would be of shape 4Nx4N.
    But in this case we keep the continuum fixed=1, so popt is of length 1 + 3xN and pcov is of shape (1+3N)x(1+3N)
    In order to generalize the output array shape (the user may not always ask for continuum to be fixed, but returning
    arrays if different shape for different cases makes it difficult to implement other parts of the code) we have to
    force it to become of length 4N. We do it by inserting the value of continuum after every 3rd element in the popt array
    and similarly for pcov array. This is less trivial than it sounds.
    '''
    if extend: n -= 1
    for yy in range(0, n):
        if extend:
            # print popt, pos+4*yy #Debugging
            popt = np.insert(popt, pos + 4 * (yy + 1), popt[popt_insert])
            pcov = np.insert(np.insert(pcov, pos + 4 * (yy + 1), pcov[pcov_insert], axis=0), pos + 4 * (yy + 1),
                             np.insert(pcov[:, pos], pos + 4 * (yy + 1), pcov[pcov_insert][pcov_insert]), axis=1)
        else:
            popt = np.insert(popt, pos + 4 * yy, popt_insert)
            pcov = np.insert(np.insert(pcov, pos + 4 * yy, pcov_insert, axis=0), pos + 4 * yy, pcov_insert, axis=1)
    return popt, pcov


# -------------Fucntion for deciding in a line is detected or not----------------------------
def isdetect(EW_signi, f_signi, f_SNR, EW_thresh=None, f_thresh=None, f_SNR_thresh=None):
    if EW_thresh is not None:
        if EW_signi >= EW_thresh:
            if f_SNR_thresh is not None:  # will check if f_SNR is above threshold if threshold is present
                if f_SNR >= f_SNR_thresh:
                    return True
                else:
                    return False
            else:
                return True
        else:
            return False
    elif f_thresh is not None:
        if f_signi >= f_thresh:
            if f_SNR_thresh is not None:  # will check if f_SNR is above threshold if threshold is present
                if f_SNR >= f_SNR_thresh:
                    return True
                else:
                    return False
            else:
                return True
        else:
            return False
    else:
        return isdetect(EW_signi, f_signi, f_SNR,
                        EW_thresh=3.)  # if neither f thresholds specified, re-calling isdetect() with a default EW_thresh=3


# -------------Fucntion for fitting any number of Gaussians----------------------------
def fit(sptemp, l, resoln, dresoln, args):
    '''
    This function fits multiple Gaussian profiles to a group of *neighbouring* lines simultaneously
    INPUTS:
    sptemp: pandas dataframe (can be thought of as a python object) containing the input spectra i.e. wavelength, flambda, flambda uncertainty
    l: pandas dataframe containing lines to be fit, includes info about vacuum line centers, initial guess of redshift, line type (emission/absorption)
    resoln: resolution of spectra R=lambda/delta_lambda
    dresoln: uncertainty in the above quantity
    args: keyword arguments to give user the control to choose from options e.g. what kind of fit, how much vel width to allow for, etc.
    '''
    global line_type_dic
    types = []
    for v in l.type.values:
        types.append(line_type_dic[v])
    # fitting 3 parameters, keeping center fixed
    if args.fix_cen:
        p_init, lbound, ubound = [sptemp.flam.values[0]], [-np.inf], [np.inf]
        for xx in range(0, len(l)):
            fl = sptemp[sptemp['wave'] >= l.wave.values[xx]].flam.values[0] - cont
            p_init = np.append(p_init, [np.abs(fl) * types[xx], l.wave.values[xx] * 2. * gf2s / resoln])
            lbound = np.append(lbound, [-np.inf if float(types[xx] < 0) else 0.,
                                        l.wave.values[xx] * 1. * gf2s / (resoln - 3. * dresoln)])
            ubound = np.append(ubound, [np.inf if float(types[xx] > 0) else 0.,
                                        l.wave.values[xx] * args.v_maxwidth * gf2s / 3e5])
        popt, pcov = curve_fit(lambda x, *p: s.fixcen_gaus(x, l.wave.values, *p), sptemp['wave'], sptemp['flam'],
                               p0=p_init, sigma=sptemp['flam_u'], absolute_sigma=True, bounds=(lbound, ubound))
        popt, pcov = update_p(popt, pcov, 2, len(l), l.wave.values, 0.)

    # fitting 3 parameters, keeping continuum fixed
    elif args.fix_cont:
        if args.fixgroupz:
            # -----if redshift of a group of lines is to be fixed----------
            p_init, lbound, ubound = [l.zz.values[0]], [0.], [np.inf]
            cont = 1.
            for xx in range(0, len(l)):  # iterating on number of lines to be fit in the input chunk of spectra
                if 'Ly-alpha' in l.label.values[xx]:
                    zz_allow = (2000. / 3e5) * (1 + l.zz.values[xx]) + 3. * l.zz_err.values[
                        xx]  # increasing allowance of z (by 2000 km/s) if the line to be fit is Ly-alpha
                elif 'MgII' in l.label.values[xx]:
                    zz_allow = (300. / 3e5) * (1 + l.zz.values[xx]) + 3. * l.zz_err.values[
                        xx]  # increasing allowance of z (by 300 km/s) if the line to be fit is MgII2797 lines, due to wind
                else:
                    zz_allow = 3. * l.zz_err.values[
                        xx]  # for all other lines, allowing the fitted redshift to vary within 3sigma error of the input initial guess of redshift
                if args.debug: print 'Deb340:', l.label.values[xx], zz_allow  # args.debug = debugging statements, ignore these henceforth
                fl = sptemp[sptemp['wave'] >= l.wave.values[xx]].flam.values[0] - cont  # initial guess of the height of the gaussian = flambda value at the line center - continuum (cont)

                p_init = np.append(p_init, [np.abs(fl) * types[xx], l.wave.values[xx] * np.mean(
                    [1. * gf2s / (resoln - 3. * dresoln),
                     args.v_maxwidth * gf2s / 3e5])])  # initial guess for each line
                lbound = np.append(lbound, [-np.inf if float(types[xx] < 0) else 0., l.wave.values[xx] * 1. * gf2s / (
                        resoln + 3. * dresoln)])  # lower bounds of parameters for each line
                ubound = np.append(ubound, [np.inf if float(types[xx] > 0) else 0., l.wave.values[
                    xx] * args.v_maxwidth * gf2s / 3e5])  # upper bounds of parameters for each line
                if args.debug:
                    print 'Deb345: rest frame', l.label.values[xx], l.wave.values[xx] / (1 + l.zz.values[xx]), lbound[1 + 3 * xx] / (1 +l.zz.values[xx]), ubound[1 + 3 * xx] / (1 + l.zz.values[xx])  #
                    print 'Deb346: obs frame', l.label.values[xx], l.wave.values[xx], lbound[1 + 3 * xx], ubound[1 + 3 * xx]  #
            popt, pcov = curve_fit(
                lambda x, *p: s.fixcont_fixgroupz_gaus(x, cont, len(l), l.wave.values, l.zz.values, *p), sptemp['wave'],
                sptemp['flam'], p0=p_init, sigma=sptemp['flam_u'], absolute_sigma=True, bounds=(lbound, ubound),
                maxfev=1500)  # to fix redshift for each group
            # popt contains the best fit parameters, pcov is covariance matrix, used to estimate uncertainties in fitted parameters
            if args.debug: print 'Deb348:', 'RMS=', np.sqrt(np.sum((s.fixcont_fixgroupz_gaus(sptemp.wave, cont, len(l),
                                                                                             l.wave.values, l.zz.values,
                                                                                             *popt) - sptemp.flam) ** 2) / len(
                sptemp))  #

            # -----The following section does something similar to update_popt() function. Refer to that for documentation
            popt[0], popt[1] = popt[1], popt[
                0]  # swapping 1st and 2nd element to preserve order: flux1, redshift, sigma1, flux2, sigma2, ....
            pcov[[0, 1]] = pcov[[1, 0]]
            pcov[:, [0, 1]] = pcov[:, [1, 0]]

            for yy in range(1, len(l)):
                popt = np.insert(popt, 1 + 3 * yy, l.wave.values[yy] * (1. + popt[1]) / (1. + l.zz.values[xx]))
                pcov = np.insert(
                    np.insert(pcov, 1 + 3 * yy, l.wave.values[yy] * (1. + pcov[1]) / (1. + l.zz.values[xx]), axis=0),
                    1 + 3 * yy,
                    np.insert(pcov[:, 1], 1 + 3 * yy, l.wave.values[yy] * (1. + pcov[1][1]) / (1. + l.zz.values[xx])),
                    axis=1)

            popt[1] = l.wave.values[0] * (1. + popt[1]) / (1. + l.zz.values[
                0])  # because first one is 'modified', no need to 'insert', the subsequent ones need to be inserted
            pcov[:, 1] = l.wave.values[0] * pcov[:, 1] / (1. + l.zz.values[0])
            pcov[1] = l.wave.values[0] * pcov[1] / (1. + l.zz.values[0])
            # ------------------------------------------------------------------------
        else:
            # -----else i.e. letting redshift of each line vary independently of another's----------
            p_init, lbound, ubound = [], [], []
            cont = 1.
            for xx in range(0, len(l)):  # iterating on number of lines to be fit in the input chunk of spectra
                if 'Ly-alpha' in l.label.values[xx]:
                    zz_allow = (2000. / 3e5) * (1 + l.zz.values[xx]) + 3. * l.zz_err.values[
                        xx]  # increasing allowance of z (by 2000 km/s) if the line to be fit is Ly-alpha
                elif 'MgII' in l.label.values[xx]:
                    zz_allow = (300. / 3e5) * (1 + l.zz.values[xx]) + 3. * l.zz_err.values[
                        xx]  # increasing allowance of z (by 300 km/s) if the line to be fit is MgII2797 lines, due to wind
                else:
                    zz_allow = 3. * l.zz_err.values[
                        xx]  # for all other lines, allowing the fitted redshift to vary within 3sigma error of the input initial guess of redshift
                if args.debug: print 'Deb340:', l.label.values[xx], zz_allow  #
                fl = sptemp[sptemp['wave'] >= l.wave.values[xx]].flam.values[
                         0] - cont  # initial guess of the height of the gaussian = flambda value at the line center - continuum (cont)
                p_init = np.append(p_init, [np.abs(fl) * types[xx], l.wave.values[xx], l.wave.values[xx] * np.mean(
                    [1. * gf2s / (resoln - 3. * dresoln),
                     args.v_maxwidth * gf2s / 3e5])])  # initial guess for each line
                lbound = np.append(lbound, [-np.inf if float(types[xx] < 0) else 0.,
                                            l.wave.values[xx] * (1. - zz_allow / (1. + l.zz.values[xx])),
                                            l.wave.values[xx] * 1. * gf2s / (
                                                    resoln + 3. * dresoln)])  # lower bounds of parameters for each line
                ubound = np.append(ubound, [np.inf if float(types[xx] > 0) else 0.,
                                            l.wave.values[xx] * (1. + zz_allow / (1. + l.zz.values[xx])), l.wave.values[
                                                xx] * args.v_maxwidth * gf2s / 3e5])  # upper bounds of parameters for each line
                if args.debug:
                    print 'Deb345: rest frame', l.label.values[xx], l.wave.values[xx] / (1 + l.zz.values[xx]), lbound[
                                                                                                                   1 + 3 * xx] / (
                                                                                                                       1 +
                                                                                                                       l.zz.values[
                                                                                                                           xx]), \
                    ubound[1 + 3 * xx] / (1 + l.zz.values[xx])  #
                    print 'Deb346: obs frame', l.label.values[xx], l.wave.values[xx], lbound[1 + 3 * xx], ubound[
                        1 + 3 * xx]  #

            popt, pcov = curve_fit(lambda x, *p: s.fixcont_gaus(x, cont, len(l), *p), sptemp['wave'], sptemp['flam'],
                                   p0=p_init, sigma=sptemp['flam_u'], absolute_sigma=True,
                                   bounds=(lbound, ubound))  # actual fitting part
            # popt contains the best fit parameters, pcov is covariance matrix, used to estimate uncertainties in fitted parameters
            if args.debug: print 'Deb348:', 'RMS=', np.sqrt(
                np.sum((s.fixcont_gaus(sptemp.wave, cont, len(l), *popt) - sptemp.flam) ** 2) / len(sptemp))  #
            # ----------------------------------------------------
        popt, pcov = update_p(popt, pcov, 0, len(l), cont, 0.)  # to generalise output array shape
    # fitting all 4 parameters, nothing fixed
    else:
        p_init, lbound, ubound = [ly[0]], [-np.inf], [np.inf]
        for xx in range(0, len(l)):
            zz_allow = (2000. / 3e5) + 3. * l.zz_err.values[xx] if 'Ly-alpha' in l.label.values[xx] else 3. * \
                                                                                                         l.zz_err.values[
                                                                                                             xx]  # increasing allowance of z (by 2000 km/s) if the line to be fit is Ly-alpha
            fl = sptemp[sptemp['wave'] >= l.wave.values[xx]].flam.values[0] - cont
            p_init = np.append(p_init,
                               [np.abs(fl) * types[xx], l.wave.values[xx], l.wave.values[xx] * 2. * gf2s / resoln])
            lbound = np.append(lbound, [-np.inf if float(types[xx] < 0) else 0.,
                                        l.wave.values[xx] * (1. - zz_allow / (1. + l.zz.values[xx])),
                                        l.wave.values[xx] * 1. * gf2s / (resoln - 3. * dresoln)])
            ubound = np.append(ubound, [np.inf if float(types[xx] > 0) else 0.,
                                        l.wave.values[xx] * (1. + zz_allow / (1. + l.zz.values[xx])),
                                        l.wave.values[xx] * args.v_maxwidth * gf2s / 3e5])
        popt, pcov = curve_fit(lambda x, *p: s.gaus(x, len(l), *p), sptemp['wave'], sptemp['flam'], p0=p_init,
                               sigma=sptemp['flam_u'], absolute_sigma=True, bounds=(lbound, ubound))
        popt, pcov = update_p(popt, pcov, 0, len(l), 0, 0, extend=True)
    return popt, pcov


# -------------Fucntion for fitting CII2323-8 group of lines with special constraints----------------------------
def fit_CII2323_group(sptemp, l, resoln, dresoln, args):
    print 'Special treatment for fitting CII2323-8 group'

    ratio_dict = { \
        'OIII2320': 0.00024, \
        'CII2323': 0.09618, \
        'CII2325b': 0.15975, \
        'CII2325c': 1.00000, \
        'CII2325d': 0.58134, \
        'CII2328': 0.18092, \
        'SiII2335a': 0.01162, \
        'SiII2335b': 0.05834 \
        }

    def n_fixcont_gaus(x, cont, n, ratios, *p):
        result = cont
        for xx in range(n):
            result += p[0] * ratios[xx] * exp(-((x - p[2 * xx + 1]) ** 2) / (2 * p[2 * xx + 2] ** 2))
        return result

    def n_fixcont_fixgroupz_gaus(x, cont, n, ratios, obswave, zz, *p):
        result = cont
        for xx in range(n):
            result += p[0] * ratios[xx] * exp(
                -((x - (obswave[xx] * (1. + p[1]) / (1. + zz[xx]))) ** 2) / (2 * p[xx + 2] ** 2))
        return result

    # fitting 1 parameters for flux (and scaling the rest), 1 parameter for redshift and \
    # 1 parameter each for line widths, keeping continuum
    if args.fix_cont:
        if args.fixgroupz:
            cont = 1.
            norm_ind = np.where(l.label.values == 'CII2325c')[0][
                0]  # index of the line with respect to which ratios have been normalized
            fl = sptemp[sptemp['wave'] >= l.wave.values[norm_ind]].flam.values[0] - cont
            p_init, lbound, ubound, ratios = [fl, l.zz.values[0]], [0., 0.], [np.inf, np.inf], []
            for xx in range(len(l)):
                zz_allow = 3. * l.zz_err.values[xx]
                ratios.append(ratio_dict[l.label.values[xx]])
                p_init = np.append(p_init, [l.wave.values[xx] * np.mean(
                    [1. * gf2s / (resoln - 3. * dresoln), args.v_maxwidth * gf2s / 3e5])])  # 2.*gf2s/resoln])
                lbound = np.append(lbound, [l.wave.values[xx] * 1. * gf2s / (resoln + 3. * dresoln)])
                ubound = np.append(ubound, [l.wave.values[xx] * args.v_maxwidth * gf2s / 3e5])

            popt, pcov = curve_fit(
                lambda x, *p: n_fixcont_fixgroupz_gaus(x, cont, len(l), ratios, l.wave.values, l.zz.values, *p),
                sptemp['wave'], sptemp['flam'], p0=p_init, sigma=sptemp['flam_u'], absolute_sigma=True,
                bounds=(lbound, ubound))

            # ----to account for redshifts--------
            for yy in range(1, len(l)):
                popt = np.insert(popt, 1 + 2 * yy, l.wave.values[yy] * (1. + popt[1]) / (1. + l.zz.values[xx]))
                pcov = np.insert(
                    np.insert(pcov, 1 + 2 * yy, l.wave.values[yy] * (1. + pcov[1]) / (1. + l.zz.values[xx]), axis=0),
                    1 + 2 * yy,
                    np.insert(pcov[:, 1], 1 + 2 * yy, l.wave.values[yy] * (1. + pcov[1][1]) / (1. + l.zz.values[xx])),
                    axis=1)

            popt[1] = l.wave.values[0] * (1. + popt[1]) / (1. + l.zz.values[
                0])  # because first one is 'modified', no need to 'insert', the subsequent ones need to be inserted
            pcov[:, 1] = l.wave.values[0] * pcov[:, 1] / (1. + l.zz.values[0])
            pcov[1] = l.wave.values[0] * pcov[1] / (1. + l.zz.values[0])
        else:
            cont = 1.
            norm_ind = np.where(l.label.values == 'CII2325c')[0][
                0]  # index of the line with respect to which ratios have been normalized
            fl = sptemp[sptemp['wave'] >= l.wave.values[norm_ind]].flam.values[0] - cont
            p_init, lbound, ubound, ratios = [fl], [0.], [np.inf], []
            for xx in range(len(l)):
                zz_allow = 3. * l.zz_err.values[xx]
                ratios.append(ratio_dict[l.label.values[xx]])
                p_init = np.append(p_init, [l.wave.values[xx], l.wave.values[xx] * np.mean(
                    [1. * gf2s / (resoln - 3. * dresoln), args.v_maxwidth * gf2s / 3e5])])  # 2.*gf2s/resoln])
                lbound = np.append(lbound, [l.wave.values[xx] * (1. - zz_allow / (1. + l.zz.values[xx])),
                                            l.wave.values[xx] * 1. * gf2s / (resoln + 3. * dresoln)])
                ubound = np.append(ubound, [l.wave.values[xx] * (1. + zz_allow / (1. + l.zz.values[xx])),
                                            l.wave.values[xx] * args.v_maxwidth * gf2s / 3e5])

            popt, pcov = curve_fit(lambda x, *p: n_fixcont_gaus(x, cont, len(l), ratios, *p), sptemp['wave'],
                                   sptemp['flam'], p0=p_init, sigma=sptemp['flam_u'], absolute_sigma=True,
                                   bounds=(lbound, ubound))

        # ----to account for flux ratios--------
        popt[0] *= ratios[0]  # because we will be dividing by ratios[0] later
        pcov[:, 0] *= ratios[0]
        pcov[0] *= ratios[0]
        pcov[0][0] /= ratios[0]  # to account for the over-multiplication for [0][0]th cell, in the previous two steps

        for yy in range(1, len(l)):
            popt = np.insert(popt, 0 + 3 * yy, popt[0] * ratios[yy] / ratios[0])
            pcov = np.insert(np.insert(pcov, 0 + 3 * yy, pcov[0] * ratios[yy] / ratios[0], axis=0), 0 + 3 * yy,
                             np.insert(pcov[:, 0], 0 + 3 * yy, pcov[0][0] * ratios[yy] / ratios[0]), axis=1)
            # ----------------------------------------
        popt, pcov = update_p(popt, pcov, 0, len(l), cont, 0.)  # to account for continuum values

    else:
        print "Special treatment of CII2323-8 group available only in fix_cont mode. Exiting."
        sys.exit()
    return popt, pcov


# -------------Function for updating the linelist data frame by adding each line------------
def update_dataframe(sp, label, l, df, resoln, dresoln, popt=None, pcov=None, fit_successful=True, EW_thresh=None,
                     f_thresh=None, f_SNR_thresh=None):
    global line_type_dic
    # ------calculating EW using simple summation------
    wv, f, f_c, f_u = s.cutspec(sp.wave, sp.flam * sp.flam_autocont, sp.flam_autocont, sp.flam_u * sp.flam_autocont,
                                l.wave * (1. - 2. * gs2f / (resoln - dresoln)),
                                l.wave * (1. + 2. * gs2f / (resoln - dresoln)))  # +/- 2 sigma ->FWHM
    disp = [j - i for i, j in zip(wv[:-1], wv[1:])]
    EWr_sum, EWr_sum_u = np.array(
        jrr.spec.calc_EW(f[:-1], f_u[:-1], f_c[:-1], 0., disp, l.zz))  # *1.05 #aperture correction sort of
    # -------3sigma limit from Schneider et al. 1993 prescription------------
    sign = (line_type_dic[l.type])  # to take care of emission/absorption
    EWr_3sig_lim = -1. * sign * 3. * (sp.loc[sp.wave >= l.wave].W_u_interp.values[0]) / (
            1 + l.zz)  # dividing by (1+z) as the dataframe has observed frame EW limits
    fl_3sig_lim = -1. * EWr_3sig_lim * sp.loc[sp.wave >= l.wave].flam_autocont.values[0]
    # --------------------------------------------------
    if fit_successful:
        cont = sp.loc[sp.wave >= popt[2]].flam_autocont.values[0]  # continuum value at the line centre
        # Let a=continuum, b=height, c= mean, d=width of each gaussian. Then,
        # EW = constant * b*d/a. Assuming variance_aa = vaa and so on,
        # var_EW = (d/a)^2*vbb + (b/a)^2*vdd + (bd/a^2)^2*vaa + 2(bd/a^2)*(vbd - (d/a)*vba - (b/a)vda) * constant
        EWr_fit = np.sqrt(2 * np.pi) * (-1.) * popt[1] * popt[3] / (
                popt[0] * (1. + l.zz))  # convention: -ve EW is EMISSION
        EWr_fit_u = np.sqrt(2 * np.pi * (
                pcov[1][1] * (popt[3] / popt[0]) ** 2 + pcov[3][3] * (popt[1] / popt[0]) ** 2 + pcov[0][0] * (
                popt[1] * popt[3] / popt[0] ** 2) ** 2 + 2 * (popt[1] * popt[3] / popt[0] ** 2) * (
                        pcov[1][3] - (popt[3] / popt[0]) * pcov[1][0] - (popt[1] / popt[0]) * pcov[3][0]))) / (
                            1. + l.zz)
        zz = popt[2] * (1. + l.zz) / l.wave - 1.
        zz_u = np.sqrt(pcov[2][2]) * (1. + l.zz) / l.wave
        # f = constant * b*d,
        # var_f = d^2*vbb + b^2*vdd + 2b*d*vbd * constant
        f_line = np.sqrt(2 * np.pi) * popt[1] * popt[3] * cont  # total flux = integral of guassian fit
        f_line_u = np.sqrt(2 * np.pi * (
                pcov[1][1] * popt[3] ** 2 + pcov[3][3] * popt[1] ** 2 + 2 * popt[1] * popt[3] * pcov[1][
            3])) * cont  # multiplied with cont at that point in wavelength to get units back in ergs/s/cm^2
        EW_signi = 3. * EWr_fit / EWr_3sig_lim  # computing significance of detection in EW
        f_signi = 3. * f_line / fl_3sig_lim  # computing significance of detection in flux
        f_SNR = f_line / f_line_u
        detection = isdetect(EW_signi, f_signi, f_SNR, EW_thresh=EW_thresh, f_thresh=f_thresh,
                             f_SNR_thresh=f_SNR_thresh)
        # this is where all the parameters of a measured line is put into the final dataframe (called line_table in EW_fitter.py)
        # please note that variables EWr_3sig_lim and fl_3sig_lim here are referred to as Ewr_Suplim and f_Suplim respectively, in EW_fitter.py
        row = np.array([label, l.label, ("%.4f" % l.wave), ("%.4f" % float(l.wave / (1 + l.zz))), l.type, \
                        ("%.4f" % EWr_fit), ("%.4f" % EWr_fit_u), ("%.4f" % EWr_sum), ("%.4f" % EWr_sum_u),
                        ("%.4e" % f_line), \
                        ("%.4e" % f_line_u), ("%.4f" % EWr_3sig_lim), ("%.3f" % EW_signi), ("%.4e" % fl_3sig_lim),
                        ("%.3f" % f_signi), ("%.4f" % popt[0]), ("%.4f" % popt[1]), \
                        ("%.4f" % popt[2]), ("%.4f" % np.sqrt(pcov[2][2])), ("%.4f" % popt[3]), ("%.5f" % zz),
                        ("%.5f" % zz_u)])
    else:
        detection = False
        row = np.array(
            [label, l.label, ("%.4f" % l.wave), ("%.4f" % float(l.wave / (1 + l.zz))), l.type, np.nan, np.nan, \
             ("%.4f" % EWr_sum), ("%.4f" % EWr_sum_u), np.nan, np.nan, ("%.4f" % EWr_3sig_lim), \
             ("%.4e" % fl_3sig_lim), np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
    df.loc[len(df)] = row
    return detection


# -------Function to plot the gaussians-------------
def plot_gaus(sptemp, popt, cen, label, zz, tot_fl, args, detection=True):
    if args.debug: print 'Deb458:', label, cen, popt  #
    gauss_curve_lam = np.multiply(s.gaus1(sptemp.wave, *popt), sptemp.flam_autocont)
    gauss_curve_nu = jrr.spec.flam2fnu(sptemp.wave, gauss_curve_lam)
    if detection:
        if not args.plotfnu:
            plt.plot(sptemp.wave, gauss_curve_lam/args.const_for_plot, color='saddlebrown', linewidth=1, linestyle='-')
        else:
            plt.plot(sptemp.wave, gauss_curve_nu/args.const_for_plot, color='saddlebrown', linewidth=1, linestyle='-')
        plt.axvline(popt[2], c='saddlebrown', lw=0.5)
        plt.text(popt[2] + 0.1, plt.gca().get_ylim()[-1] * 0.9, label, color='saddlebrown', rotation=90, size='x-small')
        if not args.silent: print 'Detected', label
    else:
        if not args.plotfnu:
            plt.plot(sptemp.wave, gauss_curve_lam/args.const_for_plot, color='k', linewidth=1, linestyle='--')
        else:
            plt.plot(sptemp.wave, gauss_curve_nu/args.const_for_plot, color='k', linewidth=1, linestyle='--')
        plt.axvline(popt[2], c='k', lw=1)
        plt.text(popt[2] + 0.1, plt.gca().get_ylim()[-1] * 0.9, label, color='k', rotation=90, size='x-small')
        if not args.silent: print 'NOT detected', label
    #plt.axvline(cen, c='blue', lw=0.5) # comment out to NOT plot the initial guess
    if not args.plotfnu:
        tot_fl += gauss_curve_lam
    else:
        tot_fl += gauss_curve_nu
    return tot_fl


# ---------returns a line flux value based on atomic line ratios and other detected lines of same species-------
def get_flux_from_atomic(line_table, labelcol='line_label', fluxcol='flux', fluxucol='flux_u', wavecol='rest_wave',
                         dered_fluxcol='flux_dered', dered_fluxucol='flux_redcor_u', notescol='Notes', bad_value='-'):
    atomic_ratio_lines = {'OIII1660': 'OIII1666', 'NII6549': 'NII6584'}
    ratios = {'OIII1660/OIII1666': 0.34147, 'NII6549/NII6584': 0.339878}
    rest_waves = {'OIII1660': 1660.809, 'NII6549': 6549.861}
    # ---to replace bad/non detections based on ratio----------
    bad_lines = line_table[line_table[fluxucol] == bad_value].reset_index(drop=True)  # lines with upper limits
    good_lines = line_table[line_table[fluxucol] != bad_value].reset_index(drop=True)  # all the rest
    # print 'Trying to replace undetected line fluxes based on ratios from atomic physics..'
    # print str(len(bad_lines))+' bad lines found, out of which '+str(np.sum([bad_lines[labelcol].values[ind] in atomic_ratio_lines for ind in range(len(bad_lines))]))+' can be replaced (are in the replace-list).'
    for i in range(len(bad_lines)):
        thislabel = bad_lines.loc[i][labelcol]
        if thislabel in atomic_ratio_lines and atomic_ratio_lines[thislabel] in line_table[labelcol].values:
            partner_line_label = atomic_ratio_lines[thislabel]
            partner_line_u = line_table[line_table[labelcol] == partner_line_label][fluxucol].values[0]
            if partner_line_u != '-' and partner_line_u != 'nan':
                bad_lines.ix[i, fluxcol] = '%.2F' % (ratios[thislabel + '/' + partner_line_label] * float(
                    line_table[line_table[labelcol] == partner_line_label][fluxcol].values[0]))
                bad_lines.ix[i, fluxucol] = '%.2F' % (
                        ratios[thislabel + '/' + partner_line_label] * float(partner_line_u))
                if dered_fluxcol in line_table:
                    bad_lines.ix[i, dered_fluxcol] = '%.2F' % (ratios[thislabel + '/' + partner_line_label] * float(
                        line_table[line_table[labelcol] == partner_line_label][dered_fluxcol].values[0]))
                    bad_lines.ix[i, dered_fluxucol] = '%.2F' % (ratios[thislabel + '/' + partner_line_label] * float(
                        line_table[line_table[labelcol] == partner_line_label][dered_fluxucol].values[0]))
                if notescol in line_table: bad_lines.ix[
                    i, notescol] = 'Tied~to~' + partner_line_label + '~by~atomic~ratio'
                print thislabel, 'tied to', partner_line_label, 'by atomic ratio'
    line_table = pd.concat([good_lines, bad_lines], ignore_index=True).sort_values(wavecol).reset_index(drop=True)
    # ------to add to the table in case it wasn't present in the first place--

    for thislabel in atomic_ratio_lines:
        if thislabel not in line_table[labelcol].values and atomic_ratio_lines[thislabel] in line_table[
            labelcol].values:
            index = line_table[labelcol] == atomic_ratio_lines[thislabel]
            ratio = ratios[thislabel + '/' + atomic_ratio_lines[thislabel]]
            row = {labelcol: thislabel, wavecol: rest_waves[thislabel],
                   fluxcol: ratio * line_table[index][fluxcol].values[0], fluxucol: \
                       ratio * line_table[index][fluxucol].values[0]}
            if dered_fluxcol in line_table:
                row[dered_fluxcol] = ratio * line_table[index][dered_fluxcol].values[0]
                row[dered_fluxucol] = ratio * line_table[index][dered_fluxucol].values[0]
            line_table = line_table.append(row, ignore_index=True)
            print thislabel, 'tied to', atomic_ratio_lines[thislabel], 'by atomic ratio'
    return line_table


# -------Functions to correct for extinction for rcs0327-E ONLY-------------
'''
#----From http://webast.ast.obs-mip.fr/hyperz/hyperz_manual1/node10.html website using Calzetti 2000 law----
def kappa(w, i):
    if i==0:
        k = 0
    elif i==1:
        k = 2.659*(-2.156+1.509/w-0.198/(w**2)+0.011/(w**3)) + Rv
    elif i==2:
        k = 2.659*(-1.857 + 1.040/w) + Rv
    return k
#------------------Function to calculate extinction and de-redden fluxes-------------
def extinct(wave, flux, flux_u, E, E_u, inAngstrom=True):
    if inAngstrom: wave/=1e4 #to convert to micron
    wbreaks = [0.12, 0.63, 2.2] #for Calzetti 2000 law
    
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
'''


# ----From Clayton Cardelli Mathis 1989 law----
def kappa(x, i):
    Rv = 3.1  # Clayton Cardelli Mathis 1989
    x = np.array(x)
    if i == 1:
        a = 0.574 * x ** 1.61
        b = -0.527 * x ** 1.61
    elif i == 2:
        y = x - 1.82
        a = 1 + 0.17699 * y - 0.50447 * y ** 2 - 0.02427 * y ** 3 + 0.72085 * y ** 4 + 0.01979 * y ** 5 - 0.77530 * y ** 6 + 0.32999 * y ** 7
        b = 1.41338 * y + 2.28305 * y ** 2 + 1.07233 * y ** 3 - 5.38434 * y ** 4 - 0.62251 * y ** 5 + 5.30260 * y ** 6 - 2.09002 * y ** 7
    elif i == 3:
        a = 1.752 - 0.316 * x - 0.104 / ((x - 4.67) ** 2 + 0.341)
        b = -3.090 + 1.825 * x + 1.206 / ((x - 4.62) ** 2 + 0.263)
    elif i == 4:
        a = 1.752 - 0.316 * x - 0.104 / ((x - 4.67) ** 2 + 0.341) - 0.04473 * (x - 5.9) ** 2 - 0.009779 * (x - 5.9) ** 3
        b = -3.090 + 1.825 * x + 1.206 / ((x - 4.62) ** 2 + 0.263) + 0.2130 * (x - 5.9) ** 2 - 0.1207 * (x - 5.9) ** 3
    elif i == 5:
        a = -1.073 - 0.628 * (x - 8) + 0.137 * (x - 8) ** 2 - 0.070 * (x - 8) ** 3
        b = 13.670 + 4.257 * (x - 8) - 0.420 * (x - 8) ** 2 + 0.374 * (x - 8) ** 3
    return a * Rv + b


# ------------------calculate kappa for full wavelength range-----
def getfullkappa(wave, inAngstrom=True):
    flag = 0
    if type(wave) in [float, int, np.float64]:
        wave = [float(wave)]
        flag = 1
    wave = np.array(wave)
    if inAngstrom: wave /= 1e4  # to convert to micron
    x = 1. / wave
    Rv = 3.1
    k = np.zeros(len(x))
    k += kappa(x, 1) * ((x >= 0.3) & (x <= 1.1))
    k += kappa(x, 2) * ((x > 1.1) & (x <= 3.3))
    k += kappa(x, 3) * ((x > 3.3) & (x < 5.9))
    k += kappa(x, 4) * ((x >= 5.9) & (x <= 8.))
    k += kappa(x, 5) * ((x >= 8.) & (x <= 10.))
    if flag: k = k[0]  # if a single wavelength was input as a float, output a float (not array)
    return k


# ------------------Function to calculate extinction and de-redden fluxes-------------
def extinct(wave, flux, flux_u, E, E_u, inAngstrom=True, doMC=True, size=1000):
    k = getfullkappa(wave, inAngstrom=inAngstrom)
    if doMC:
        print 'Calculating de-redenned fluxes and errors via MCMC with ' + str(size) + ' iterations...'
        flux_redcor_arr = []
        for iter in range(size):
            # print 'Doing iter', iter, 'of', size #
            flux_redcor_iter = []
            for i in range(len(flux)):
                f_iter = np.random.normal(loc=flux[i], scale=flux_u[i]) if flux_u[i] > 0. else flux[i]
                E_iter = np.random.normal(loc=E, scale=E_u)
                flux_redcor_iter.append(np.multiply(f_iter, 10 ** (0.4 * k[i] * E_iter)))
            flux_redcor_arr.append(flux_redcor_iter)
        flux_redcor_arr = np.array(flux_redcor_arr)
        flux_redcor = np.median(flux_redcor_arr, axis=0)
        flux_redcor_u = np.std(flux_redcor_arr, axis=0)
    else:
        print 'Calculating de-redenned fluxes and errors via mathematically propagating uncertainties.'
        flux_redcor = np.multiply(flux, 10 ** (0.4 * k * E))
        flux_redcor_u = np.multiply(flux_u, 10 ** (
                0.4 * k * E))  # multiplying flux uncertainty with dereddning factor; no error "propagation"
        # flux_redcor_u = np.multiply(10**(0.4*k*E),np.sqrt(flux_u**2 + (flux*0.4*k*np.log(10)*E_u)**2)) #error propagation
    return flux_redcor, flux_redcor_u


# -------Function to calculate one sigma error in flux at certain wavelength--------
# ------------------NOT USED ANYMORE-----------------------------
def calc_1sig_err(sptemp, cen, resoln):
    dpix = sptemp.wave.values[2] - sptemp.wave.values[1]
    dlambda = 5. * cen / resoln
    sptemp = sptemp[sptemp['wave'].between(cen - dlambda, cen + dlambda)]
    try:
        err_wt_mean = np.power(np.sum(1 / ((sptemp['flam_u'] * sptemp['flam_autocont'] * dpix) ** 2. + (0.) ** 2)),
                               -0.5)
        wt_mean = np.sum((sptemp['flam'] * sptemp['flam_autocont'] * dpix) / (
                (sptemp['flam_u'] * sptemp['flam_autocont'] * dpix) ** 2.)) / (err_wt_mean ** -2.)
    except ZeroDivisionError:
        return 999
    return wt_mean, err_wt_mean


# -------Function to check if good (3-sigma) detection------
# ------------------NOT USED ANYMORE-----------------------------
def check_3sig_det(sptemp, l, popt, resoln, args=None):
    f_line = np.abs(np.sqrt(2 * np.pi) * popt[1] * popt[3]) * sptemp[sptemp.wave >= popt[2]].flam_autocont.values[0]
    wt_mean, err_wt_mean = calc_1sig_err(sptemp, l.wave, resoln)
    if f_line >= 3. * np.abs(err_wt_mean):
        return True, wt_mean, err_wt_mean
    else:
        if not args.silent:
            print 'Non detection of', l.label  #
        return False, wt_mean, err_wt_mean


# -------Function to calculate med_bin_flux and mad_bin_flux to get upper limit on detection------
# ------------------NOT USED ANYMORE-----------------------------
def calc_detec_lim(sp_orig, line, resoln, args):
    dlambda = 2. * gs2f / resoln
    leftlim = line.wave.values[0] * (1. - 5. / resoln) * (1. - dlambda)
    rightlim = line.wave.values[-1] * (1. + 5. / resoln) * (1. + dlambda)
    l_arr = np.concatenate((np.linspace(leftlim * (1. - (args.nbin - 1) * dlambda), leftlim, args.nbin),
                            np.linspace(rightlim, rightlim * (1. + (args.nbin - 1) * dlambda), args.nbin)))
    fluxes = []
    for l in l_arr:
        sptemp = sp_orig[sp_orig['wave'].between(l * (1. - dlambda),
                                                 l * (1. + dlambda))]  # this is sp_orig, hence NOT continuum normalised
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
            dpix = 0.3  # from other parts of the spectra
        fluxes.append(np.sum((sptemp['flam'] - sptemp['flam_autocont']) * dpix))
    return np.median(fluxes), stat.mad(fluxes)


# -------------Fucntion for fitting multiple lines----------------------------
def fit_some_EWs(line, sp, resoln, label, df, dresoln, sp_orig, args=None):
    # This is what Ayan needs to fill in, from his previous code.
    # Should work on sp.wave, sp.flam, sp.flam_u, sp.flam_autocont
    global line_type_dic
    line_type_dic = {'EMISSION': 1., 'FINESTR': 1., 'PHOTOSPHERE': -1., 'ISM': -1., 'WIND': -1., 'INTERVE': -1}
    if args.fcen is not None:
        args.fix_cen = int(args.fcen)
    else:
        args.fix_cen = 0
    if args.fcon is not None:
        args.fix_cont = int(args.fcon)
    else:
        args.fix_cont = 1
    if args.vmax is not None:
        args.v_maxwidth = float(args.vmax)
    else:
        args.v_maxwidth = 300.  # in km/s, to set the maximum FWHM that can be fit as a line
    if args.nbin is not None:
        args.nbin = int(args.nbin)
    else:
        args.nbin = 5
    if args.ndlambda is not None:
        args.ndlambda = float(args.ndlambda)
    else:
        args.ndlambda = 5.
    # -----------
    kk, c = 1, 0
    ndlambda_left, ndlambda_right = [args.ndlambda] * 2  # how many delta-lambda wide will the window (for line fitting) be on either side of the central wavelength, default 5
    try:
        c = 1
        first, last = [line.wave.values[0]] * 2
        frac_delta_center_first, frac_delta_center_last = [line.zz_err.values[0] / (1. + line.zz.values[
            0])] * 2  # fractional uncertainty in central obs-frame wavelength due to uncertainty in redshift
        if 'Ly-alpha' in line.label.values[0]:  # treating Ly-alpha specially: widening the wavelength window
            ndlambda_left, ndlambda_right = 3., 12.
    except IndexError:
        pass
    while kk <= len(line):
        center1 = last
        frac_delta_center1 = frac_delta_center_last  # fractional uncertainty in central obs-frame wavelength due to uncertainty in redshift
        if kk == len(line):
            center2 = 1e10  # just an insanely high number, required to plot last line
            frac_delta_center2 = 0.
        else:
            center2 = line.wave.values[kk]
            frac_delta_center2 = line.zz_err.values[kk] / (1. + line.zz.values[kk])  # fractional uncertainty in central obs-frame wavelength due to uncertainty in redshift
        if center2 * (1. - ndlambda_left / resoln - 3 * frac_delta_center2) > center1 * (1. + ndlambda_right / resoln + 3 * frac_delta_center1):
            if args.debug: print 'Deb662: before fitting', first * (
                    1. - ndlambda_left / resoln - 3 * frac_delta_center_first), first, last, last * (
                    1. + ndlambda_right / resoln + 3 * frac_delta_center_last)  #
            sp2 = sp[sp['wave'].between(first * (1. - ndlambda_left / resoln - 3 * frac_delta_center_first),
                                        last * (1. + ndlambda_right / resoln + 3 * frac_delta_center_last))]
            sp2.flam = sp2.flam / sp2.flam_autocont  # continuum normalising by autocont
            sp2.flam_u = sp2.flam_u / sp2.flam_autocont  # continuum normalising by autocont

            if not args.showbin:
                plt.axvline(np.min(sp2.wave), c='blue', lw=0.5, linestyle='--')
                plt.axvline(np.max(sp2.wave), c='blue', lw=0.5, linestyle='--')

            if not args.silent:
                print 'Trying to fit', line.label.values[kk - c:kk], 'line/s at once. Total', c
            # med_bin_flux, mad_bin_flux = calc_detec_lim(sp_orig, line[kk-c:kk], resoln, args) #NOT REQUIRED anymore
            try:
                if set(['CII2323', 'CII2325c', 'CII2325d', 'CII2328']) <= set(line[
                                                                              kk - c:kk].label.values):  # special treatment for CII2323-8 group, to fit with additional constraints of relative line ratios
                    popt, pcov = fit_CII2323_group(sp2, line[kk - c:kk], resoln, dresoln, args)
                    # popt, pcov = fit(sp2, line[kk-c:kk], resoln, dresoln, args)
                else:
                    popt, pcov = fit(sp2, line[kk - c:kk], resoln, dresoln, args)
                tot_fl = np.zeros(len(sp2))
                for xx in range(0, c):
                    ind = line.index.values[(kk - 1) - c + 1 + xx]
                    # det_3sig, wt_mn, er_wt_mn = check_3sig_det(sp2, line.loc[ind], popt[4*xx:4*(xx+1)], resoln, args=args) # check if 3 sigma detection; NOT REQUIRED anymore
                    detection = update_dataframe(sp2, label, line.loc[ind], df, resoln, dresoln,
                                                 popt=popt[4 * xx:4 * (xx + 1)],
                                                 pcov=pcov[4 * xx:4 * (xx + 1), 4 * xx:4 * (xx + 1)],
                                                 fit_successful=True)
                    tot_fl = plot_gaus(sp2, popt[4 * xx:4 * (xx + 1)], line.loc[ind].wave, line.loc[ind].label,
                                       line.loc[ind].zz, tot_fl, args=args, detection=detection)
                if c > 1:
                    if not args.plotfnu:
                        plt.plot(sp2.wave, np.subtract(tot_fl, (c - 1.) * np.multiply(popt[0], sp2.flam_autocont))/args.const_for_plot, color='darkcyan', linewidth=2)
                    else:
                        plt.plot(sp2.wave, np.subtract(tot_fl, (c - 1.) * jrr.spec.flam2fnu(sp2.wave,
                                                                                            np.multiply(popt[0],
                                                                                                        sp2.flam_autocont))),
                                 color='darkcyan', linewidth=2)
                if not args.silent:
                    print 'done above fitting'

            except (RuntimeError, ValueError, IndexError) as e:
                for xx in range(0, c):
                    ind = line.index.values[(kk - 1) - c + 1 + xx]
                    plt.axvline(line.loc[ind].wave, c='k', lw=1)
                    # wt_mn, er_wt_mn = calc_1sig_err(sp2, line.loc[ind].wave, resoln) #NOT REQUIRED anymore
                    try:
                        dummy = update_dataframe(sp2, label, line.loc[ind], df, resoln, dresoln, fit_successful=False)
                    except:
                        pass
                if not args.silent:
                    print 'Could not fit these', c, 'lines.'
                    print 'Error in ayan.mage.fit_some_EWs:', e  #

            first, last = [center2] * 2
            frac_delta_center_first, frac_delta_center_last = [frac_delta_center2] * 2
            if kk < len(line) and 'Ly-alpha' in line.label.values[
                kk]:  # treating Ly-alpha specially: widening the wavelength window
                ndlambda_left, ndlambda_right = 3., 12.
            c = 1
        else:
            last = center2
            frac_delta_center_last = frac_delta_center2
            if kk < len(line) and 'Ly-alpha' in line.label.values[kk]:  # treating Ly-alpha specially: widening the wavelength window
                ndlambda_right = 12.
            c += 1
        kk += 1

    return df  # df is a pandas data frame that has the line properties

# -------------End of functions----------------------------
