import sys
sys.path.append('/Users/acharyya/Dropbox/MagE_atlas/Tools')
import jrr
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import argparse as ap

parser = ap.ArgumentParser(description="Mage spectra fitting tool")
parser.add_argument('--n')
args, leftovers = parser.parse_known_args()
if args.n is not None:
    n = args.n
else:
    n=50

plt.close('all')
fig = plt.figure() 
fulltable = pd.read_table('/Users/acharyya/Dropbox/mage_atlas/Tools/Contrib/fitted_line_list.txt', delim_whitespace=True, comment="#")
lines = pd.read_table('/Users/acharyya/Desktop/mage_plot/emission_list.txt', delim_whitespace=True, comment="#")
excludelabel = ['S0957+0509','S1050+0017',]# 'S1429+1202']
fulltable = fulltable[~fulltable['label'].isin(excludelabel)] #
labels = np.unique(fulltable['label'])
colors = 'rcmykbgrcmykbgrcmykbg'
#lines_num = ['OIII]1660', 'OIII1666']
#lines_num = ['OIII2320', '[OIII]2331']
lines_den = ['OII2470mid']

#lines_num = ['CIII977']
#lines_num = ['CIII]1906', 'CIII]1908', 'CIII2323', 'CII2325b', 'CII2325c', 'CII2325d', 'CII2328']
#lines_den = ['CIII2323', 'CII2325b', 'CII2325c', 'CII2325d', 'CII2328']
#lines_den = ['CII1335b', 'CII1335c']

#lines_num = ['SiIII1882', 'SiIII1892']
#lines_den = ['SiII2335a', 'SiII2335b']
#lines_den = ['SiII1533']

#lines_num = ['NIV]1486']
#lines_den = ['NII1084', 'NII1085']
lines_num = ['NII]2140']

#lines_den = ['HeII1640']
ax1 = fig.add_subplot(111)
h = np.zeros(len(lines))
for ii in range(0,len(labels)):
    ew = np.arange(len(lines)+2)*np.nan
    ewu = np.arange(len(lines)+2)*np.nan
    table = fulltable[(~np.isnan(fulltable['f_line'])) & (fulltable['label'].eq(labels[ii])) & \
    (~fulltable['type'].eq('ISM')) & \
    (fulltable['significance'] > 2.)]# & \
    #(fulltable['f_line']/fulltable['f_line_u'] > 2.)]
    table = table[table['EWr_fit_u']/np.abs(table['EWr_fit']) < 3.]
    
    a, avar, b, bvar, c, cvar, d, dvar = 0.,0.,0.,0.,0.,0.,0.,0.
    for line in lines_num:
        try:
            a += table[table['line_lab'].eq(line)].EWr_fit.values[0]
            avar += table[table['line_lab'].eq(line)].EWr_fit_u.values[0]**2
        except:
            pass
    for line in lines_den:
        try:
            b += table[table['line_lab'].eq(line)].EWr_fit.values[0]
            bvar += table[table['line_lab'].eq(line)].EWr_fit_u.values[0]**2
        except:
            pass
    '''
    for line in lines_num2:
        try:
            c += table[table['line_lab'].eq(line)].EWr_fit.values[0]
            cvar += table[table['line_lab'].eq(line)].EWr_fit_u.values[0]**2
        except:
            pass
    
    for line in lines_den2:
        try:
            d += table[table['line_lab'].eq(line)].EWr_fit.values[0]
            dvar += table[table['line_lab'].eq(line)].EWr_fit_u.values[0]**2
        except:
            pass  
    '''
    if a < 0 and b < 0:
        print ii, labels[ii], a, b #
        err = jrr.util.sigma_adivb(a, np.sqrt(avar), b, np.sqrt(bvar))
        #err2 = jrr.util.sigma_adivb(c, np.sqrt(cvar), d, np.sqrt(dvar))
        pl=ax1.errorbar(b, np.divide(a,b), fmt='o', lw=0.5, xerr=np.sqrt(bvar), yerr=err)
    
    '''
    for jj, line in enumerate(lines.LineID):
        if table['line_lab'].isin([line]).any():
            #h[jj] += 1
            ew[jj+1] = table[table['line_lab'].isin([line])].EWr_fit.values[0]
            ewu[jj+1] = table[table['line_lab'].isin([line])].EWr_fit_u.values[0]
    try:
        #plt.bar(range(len(lines)), h, lw=0, align = 'center', color=colors[ii])
        pl=plt.errorbar(range(len(ew)), ew, fmt='o', lw=0.5, yerr=ewu)
        #pl=plt.errorbar(range(len(table)), table.EWr_fit.values, fmt='o', lw=0.5, yerr=table.EWr_fit_u.values)
        plt.xticks(range(len(lines)+2),np.concatenate(([' '],lines.LineID.values,[' '])), rotation = 90, fontsize='small')
        print pl[0].get_color(), table.label.values[0], len(table)    
    except:
        pass
    '''
    
    #plt.xlabel(lines_num2+['/']+lines_den2)
    plt.xlabel(lines_den)
    plt.ylabel(lines_num+['/']+lines_den)
    
    #plt.xlabel('Rest wavelength (A)')
    #plt.ylabel('Measured EW (A)')
    plt.draw()
    #plt.pause(1)

'''
ax2 = ax1.twiny()
ax2.set_xlim(ax1.get_xlim())
ax2.set_xticks(lines.restwave)
ax2.set_xticklabels(lines.LineID, rotation = 45, ha='left', fontsize='small')
'''
fig.savefig('/Users/acharyya/Desktop/plot'+str(n)+'.png')
plt.show(block=False)