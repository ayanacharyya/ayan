import numpy as np
import math
import matplotlib.pyplot as plt
import subprocess
import sys
from  astropy.io import ascii
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp
from scipy.interpolate import interp1d
from operator import itemgetter
sys.path.append('/Users/acharyya/mage/JRR')
import JRR as j
import warnings
warnings.filterwarnings("ignore")
import re
#-------------Choosing color according to line strength----------
def choosecol(n):
    if n == 1:
        return 'y'
    elif n == 2:
        return 'm'
    elif n == 3:
        return 'c'
    elif n == 4:
        return 'b'
    elif n == 5:
        return 'g'
    elif n == 6:
        return 'r'
    else:
        return 'k'
#------------Reading pre-defined line list-----------------------------
def readlist(ll, isstacked = False):
    list=[]
    fn = open(ll,'r')
    l = fn.readlines()
    for lin in l:
        if len(lin.split())>1:
            wn = lin.split()[0]
            w = lin.split()[1]
            if isfloat(w):
                if isstacked:
                    list.append([wn,float(lin.split()[2]),int(w)])
                else:
                    list.append([wn,float(w), 7])
    fn.close()
    list = sorted(list, key=itemgetter(1))
    return list
#------------Reading input spectra-------------------------------
def readspec(inp):
    wz = []
    fl = []
    fn = open(inp, 'r')
    lines = fn.readlines()
    for line in lines:
        if len(line.split())>1:
            wl = line.split()[0]
            t = line.split()[1]
            if isfloat(wl) and not np.isnan(float(t)):
                wz.append(float(wl))
                fl.append(float(t))
    return wz, fl
#-----------Reading input stacked mage spectra----------------------------
def readstackedmagespec(inp):
    s = ascii.read(inp, comment="#")
    wz = np.ma.array(s['restwave'], mask=np.isnan(s['restwave']))
    fl = np.ma.array(s['X_avg'], mask=np.isnan(s['X_avg']))
    err_f = np.ma.array(s['X_sigma'], mask=np.isnan(s['X_sigma']))
    err_f_jack = np.ma.array(s['X_jack_std'], mask=np.isnan(s['X_jack_std']))
    #err_f = np.divide(err_f, fl)
    return wz, fl, err_f, err_f_jack
#------------Read single mage spectrum---------------------------
def readsinglemagespec(inp):
    s = ascii.read(inp, format="basic", comment="#", guess=False)
    wave   = s['wave']
    wz = np.arange(np.floor(wave[0]),np.ceil(wave[-1]),0.02)#
    fl = interp1d(wave, s['fnu'], bounds_error=False)(wz)
    err_f = interp1d(wave, s['noise'], bounds_error=False)(wz)
    if re.search("wC1", inp):
        cont = interp1d(wave, s['cont_fnu'], bounds_error=False)(wz)
    else:
        cont = np.median(np.array(fl)[np.isfinite(fl)]) #1.
    err_f = np.divide(err_f, fl)
    fl = np.divide(fl, cont)
    wz = np.array(wz)[np.logical_and(np.isfinite(fl),np.logical_and(np.less(np.abs(fl),30.),np.less(np.abs(err_f),1.)))]
    err_f = np.array(err_f)[np.logical_and(np.isfinite(fl),np.logical_and(np.less(np.abs(fl),30.),np.less(np.abs(err_f),1.)))]
    fl = np.array(fl)[np.logical_and(np.isfinite(fl),np.less(np.abs(fl),30.))]
    return wz, fl, err_f
#-----------Function to check if float------------------------------
def isfloat(str):
    try: 
        float(str)
    except ValueError: 
        return False
    return True
#------------Single Gaussian function-----------------------------
def gaus1(x,*p):
    return p[0] + p[1]*exp(-((x-p[2])**2)/(2*p[3]**2))
def fixcen_gaus1(x,center,*p):
    return p[0] + p[1]*exp(-((x-center)**2)/(2*p[2]**2))
def fixcont_gaus1(x,cont,*p):
    return cont + p[0]*exp(-((x-p[1])**2)/(2*p[2]**2))
#------------Double Gaussian function (not yet working)-----------------------------
def gaus2(x,*p):
    return p[0] + p[1]*exp(-((x-p[2])**2)/(2*p[3]**2)) + p[4]*exp(-((x-p[5])**2)/(2*p[6]**2))
def fixcen_gaus2(x,center, center2, *p):
    return p[0] + p[1]*exp(-((x-center)**2)/(2*p[2]**2)) + p[3]*exp(-((x-center2)**2)/(2*p[4]**2))
def fixcont_gaus2(x,cont,*p):
    return cont + p[0]*exp(-((x-p[1])**2)/(2*p[2]**2)) + p[3]*exp(-((x-p[4])**2)/(2*p[5]**2))
#------------Multiple Gaussian function-----------------------------
def fixcont_gaus(x,cont,n,*p):
    result = cont
    for xx in range(0,n):
        result += p[3*xx+0]*exp(-((x-p[3*xx+1])**2)/(2*p[3*xx+2]**2))
    return result
def fixcen_gaus(x,list_of_cen,*p):
    result = p[0]
    for xx in range(0, len(list_of_cen)):
        result += p[2*xx+1]*exp(-((x-list_of_cen[xx])**2)/(2*p[2*xx+2]**2))
    return result
def gaus(x,n,*p):
    result = p[0]
    for xx in range(0,n):
        result += p[3*xx+1]*exp(-((x-p[3*xx+2])**2)/(2*p[3*xx+3]**2))
    return result
#-----------Continuum function------------------------------
def cont(x, *p):
    return p[0]*x**3 + p[1]*x**2 + p[2]*x + p[3]
#-------------------Function to continuum normalise--------------------------------------------
def cont_norm(lx, ly, wz, fl, c, popt, xmin, xmax, list):
    xmin = lx[0]
    xmax = lx[-1]
    wz, fl = cutspec(wz, fl, xmin, xmax)
    fl = np.divide(fl,cont(wz, *popt))
    c = 1
    print 'If you want to save this (this specific wavelength range) normalised spectrum as a file press f\n\
    NOTE: Now c has been automatically set to c = 1, so you can directly proceed with gaussian fitting as follows:\n\
    Step 1: Spot a potential line (amission/absorption) in the plot\n\
    Step 2: Left click on the left of the line you want to fit (including a fair bit of continuum)\n\
    Step 3: Left click on the right of the line (again inlcuding bit of continuum)\n\
    Step 4: Press any key (except Esc, c, z, f or s)\n\
    Repeat from Step 1 for another line if desired. The last line you fit will still be showing on the plot.'
    return wz, fl, c, xmin, xmax
#-------------Fucntion for fitting Gaussian or continuum----------------------------
def fit(lx, ly, c):
    ly = np.array(ly)
    #sigma = np.sqrt(sum(y*(x-mean)**2)/n)
    sigma = 1.
    if c == 1:
        popt,pcov = curve_fit(gaus1,lx,ly,p0=[ly[0], 1, lx[np.where(np.abs(ly-ly[0]) == np.max(np.abs(ly-ly[0])))[0][0]], sigma])
        print 'cont = '+str(format(popt[0]*const, '.3e'))+', flux = '+str(format(np.sqrt(2*np.pi)*popt[1]*popt[3]*const, '.3e'))+\
    ', center = '+str(format(popt[2], '.3f'))+'\nA, gfwhm = '+str(format(2*np.sqrt(2*np.log(2))*popt[3], '.3e'))+\
    ' A, EW = '+str(format(np.sqrt(2*np.pi)*popt[1]*popt[3]/popt[0], '.3e'))+', core = '+str(format(popt[1]*const, '.3e'))
    elif c == 2:
        popt,pcov = curve_fit(cont,lx,ly,p0=[0.1, 0.1, 0.1, np.mean(ly)])
        print 'Fitted *part* of the continuum'
    mylog.write(str(popt[0])+' '+str(popt[1])+' '+str(popt[2])+' '+str(popt[3])+'\n')
    return popt, pcov
#--------------------Fucntion to accept click--------------------------------------------
def onclick(event):
    global k, lx, ly, ax, x, c, wz, fl
    mylog.write(str(event.xdata)+' ')
    x[k%2] = event.xdata;   
    if k%2 == 1:
        w2, f2 = cutspec(wz, fl, x[0], x[1])
        if c == 1:
            lx = w2
            ly = f2
        elif c == 2:
            lx.extend(w2)
            ly.extend(f2)
        mylog.write('\n')
    k = k+1;
    xn=np.linspace(event.xdata, event.xdata,100)
    p = ax.plot(xn, y, label=str(event.xdata), linestyle='solid')
    plt.gca().add_artist(plt.legend(handles = p, loc = 2))
    plt.draw()
#-------------------Function to accept keypress--------------------------------------------
def onpress(event):
    global lx, ly, ax, popt, list, c, wz, fl, xmin2, xmax2
    print 'You pressed '+str(event.key)
    if event.key == 'c':
        print 'Sorry, <c> is not currently working, press <a> for continuum normalisation.'
        '''
        print 'Interpolating these bits of continuum...'
        compute_cont()
        print 'DONE interpolating continuum'
        '''
    elif event.key == 'a':
        print 'Continuum normalising and replotting...'
        wz, fl, c, xmin2, xmax2 = cont_norm(lx, ly, wz, fl, c, popt, xmin, xmax, list)
        makeplotint(wz, fl, xmin2, xmax2, c, list, const)
        lx=[]
        ly=[]
    elif event.key == 'f':
        outfile = inp.split()[0][0:-4]+'_cont_norm_inrange('+str(xmin2)+','+str(xmax2)+').txt'
        head  = '# Continuum normalised '+inp+' at z=1.7033 \n'
        head += '# restwave    :  Restframe wavelength, binned, in Angstroms.\n'
        head += '# flux : continuum normalised flux values \n'
        head += '# Columns are: \n'
        head += 'restwave   fnu  \n'
        np.savetxt(outfile, np.transpose([wz, fl]), "%.2f  %.2E", header=head)
        print 'Saved '+outfile
    elif event.key == 'z':
        print 'Zooming in to '+str(lx[0])+','+str(lx[-1])
        makeplotint(wz, fl, lx[0], lx[-1], c, list, const)
        lx=[]
        ly=[]
    else:
        print 'going to fit...'
        popt, pcov = fit(lx, ly, c)
        lx = np.array(lx)
        if c == 1:
            l = 'cont = '+str(format(popt[0]*const, '.3e'))+', flux = '+str(format(np.sqrt(2*np.pi)*popt[1]*popt[3]*const, '.3e'))+\
            ', center = '+str(format(popt[2], '.3f'))+'\nA, gfwhm = '+str(format(2*np.sqrt(2*np.log(2))*popt[3], '.3e'))+\
            ' A, EW = '+str(format(np.sqrt(2*np.pi)*popt[1]*popt[3]/popt[0], '.3e'))+', core = '+str(format(popt[1]*const, '.3e'))
            p = ax.plot(lx, gaus1(lx,*popt), label=l, linestyle='solid', color = 'black')
            #ax.plot(lx, gaus1(lx,np.min(ly), 1, (x[0]+x[1])/2, 1.), label='guess', linestyle='dotted')
            plt.legend(handles = p, loc =8, fontsize = 12)
        elif c == 2:
            ax.plot(lx,cont(lx, *popt), linestyle='solid', color='black')
    plt.ylim(np.min(fl)*0.8, np.max(fl)*1.2)
    plt.draw()
#-------------------Function to create the plot--------------------------------------------
def makeplot(wz, fl, xmin, xmax, list, fig, const=1, a1=1, b1=1, c1=1, er = 'NONE', er_j = 'NONE'):
    global ax
    labels=[]
    ticks=[]
    li = [a[1] for a in list]
    i = np.where(np.array(li)>xmin)[0][0] if len(np.where(np.array(li)>xmin)[0])>0 else len(np.array(li))
    j = np.where(np.array(li)<xmax)[0][-1] if len(np.where(np.array(li)<xmax)[0])>0 else -1
    #-----------------Plotting spectrum------------------------------------------------------
    ax = fig.add_subplot(int(a1),int(b1),int(c1))
    ax.step(wz, fl, color='forestgreen')
    if not er is 'NONE':
        ax.step(wz, er, color='gray')
    if not er_j is 'NONE':
        ax.step(wz, er_j, color='black')
    #----------------Labels------------------------------------------------------------------------
    ax.set_xlim([xmin,xmax])
    if er is 'NONE':
        ax.set_ylim([np.min(fl)*0.95, np.max(fl)*1.05])
    else:
        ax.set_ylim([0., min(3.,np.max(fl)*1.05)])
    fig.text(0.5, 0.04, 'Restframe Wavelength (A)', ha='center')
    fig.text(0.04, 0.5, 'f_nu (x '+str(const)+')', va='center', rotation='vertical')
    #----------------Plotting horizontal & vertical lines-------------------------------------------------
    ax.axhline(1.0, linestyle='--', color='black')
    for ii in range(i,j+1):
        ax.axvline(li[ii], ymin = (fl[np.where(wz>list[ii][1])[0][0]]-ax.get_ylim()[0])/(ax.get_ylim()[1]-ax.get_ylim()[0]),\
         color=choosecol(list[ii][2]), lw=0.5) 
    #------------------Tick labels on top-----------------------------------------------
    ax.locator_params(axis='y',nbins=6) #
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    for ii in range(i,j+1):
        ticks.append(list[ii][1])
        labels.append(list[ii][0]+str(list[ii][1]))
    ax2.set_xticks(ticks)
    ax2.set_xticklabels(labels, rotation = 45, ha='left', fontsize='small')
    return fig
#-----------Make interactive plot------------------------------
def makeplotint(wz, fl, xmin, xmax, c, list, const):
    if c == 1:
        mylog.write('#Log file for Gaussian fitting.')
    elif c == 2:
        mylog.write('#Log file for continuum fitting.')
    mylog.write(' Column format: x1 x2 p[0] p[1] p[2] p[3] where x are clicked wavelengths and p[] are fitted parameters.\n')
    plt.close('all')
    fig2 = plt.figure(figsize=(10,8))
    fig = makeplot(wz, fl, xmin, xmax, list, fig2, const=const)
    #------------------Interactive----------------------------------------------------------------------
    cid1 = fig.canvas.mpl_connect('button_press_event', onclick)
    cid2 = fig.canvas.mpl_connect('key_press_event', onpress)
    #-------------------Show/Save plot----------------------------------------------
    plt.show()
    #plt.savefig('')
#----------------Shift flux level----------------------------
def shiftflux(fl, const):
    return np.divide(fl, const)
#-----------------Shift wavelength to rest frame-----------------
def obstorest(wz, z):
    return np.divide(wz,(1.+z))
#-----------------Cut out relevant portion from spectrum----------
def cutspec(wz, fl, fl_cont, fl_u, xmin, xmax):
    fl = fl[np.where(wz>xmin)[0][0]:np.where(wz<xmax)[0][len(np.where(wz<xmax)[0])-1]+1]
    fl_cont = fl_cont[np.where(wz>xmin)[0][0]:np.where(wz<xmax)[0][len(np.where(wz<xmax)[0])-1]+1]
    fl_u = fl_u[np.where(wz>xmin)[0][0]:np.where(wz<xmax)[0][len(np.where(wz<xmax)[0])-1]+1]
    wz = wz[np.where(wz>xmin)[0][0]:np.where(wz<xmax)[0][len(np.where(wz<xmax)[0])-1]+1]
    return wz, fl, fl_cont, fl_u
def cutspec_3comp(wz, fl, fl_u, xmin, xmax):
    fl = fl[np.where(wz>xmin)[0][0]:np.where(wz<xmax)[0][len(np.where(wz<xmax)[0])-1]+1]
    fl_u = fl_u[np.where(wz>xmin)[0][0]:np.where(wz<xmax)[0][len(np.where(wz<xmax)[0])-1]+1]
    wz = wz[np.where(wz>xmin)[0][0]:np.where(wz<xmax)[0][len(np.where(wz<xmax)[0])-1]+1]
    return wz, fl, fl_u
def cutspec_2comp(wz, fl, xmin, xmax):
    fl = fl[np.where(wz>xmin)[0][0]:np.where(wz<xmax)[0][len(np.where(wz<xmax)[0])-1]+1]
    wz = wz[np.where(wz>xmin)[0][0]:np.where(wz<xmax)[0][len(np.where(wz<xmax)[0])-1]+1]
    return wz, fl
#-----------------Printing instructions------------------------------------------------
def printinstruction(c, inp, xmin, xmax):
    print 'You have opened file '+inp
    print 'You have chosen to dispplay the rest frame wavelength values in range ', xmin, 'to', xmax
    if c == 1:
        print 'You have chosen c = 1, meaning GAUSSIAN FITTING. Here are the instructions:\n\
        Step 1: Spot a potential line (amission/absorption) in the plot\n\
        Step 2: Left click on the left of the line you want to fit (including a fair bit of continuum)\n\
        Step 3: Left click on the right of the line (again inlcuding bit of continuum)\n\
        Step 4: Press any key (except Esc, c, f or s). Pressing z will zoom in the curve between your selected region.\n\
        Repeat from Step 1 for another line if desired. The last line you fit will still be showing on the plot.\n\
        NOTE: It is assumed that your spectrum is already continuum fit. If not, choose c = 2 to do continuum fitting.'
    elif c == 2:
        print 'You have chosen c = 2, meaning CONTINUUM FITTING. Here are the instructions:\n\
        Step 1: Spot a *line-less* region in the plot where you would like to fit the continuum\n\
        Step 2: Left click on the left of that region\n\
        Step 3: Left click on the right of that region\n\
        Repeat from Step 1 for another region if desired. Keep on marking multiple regions (1 region = 1 pair of click).\n\
        Step 4: Once you are done selecting all the bits of continuum, PRESS any key (except Esc, c, z, f, a or s) and wait.\n\
        CAUTION: This might take a long time depending on how many bits of continuum you have fitted already.\n\
        Step 5: In order to get continuum normalised spectra PRESS a. A new figure will pop up with normalised spectra\n\
        within a region covered by your previously selected regions.'
    else:
        print 'You have chosen c =', c, 'which does not mean anything. Please choose c = 1 or 2.\n\
        Exiting program.'
        sys.exit()
#-----------Main function starts------------------
if __name__ == '__main__':
    #-----------Declaring arrays------------------------------
    global k, lx, ly, ax, popt, x, list, c, xmin2, xmax2#, wz, fl, c,
    k = 0
    x=[0 for ii in xrange(2)]
    lx=[]
    ly=[]
    y = np.linspace(-1, 2, 100)
    #-------------Reading input file----------------------------------------------------
    fn =open('inp_splot','r')
    l=fn.readlines()[2:]
    inp = str(l[0].split()[2]) #input spectrum (text file)
    c = int(l[1].split()[2]) # choice: gaussian line fit OR continuum fit?
    xmin = float(l[2].split()[2]) # limits of wavelength which 
    xmax = float(l[3].split()[2]) # are to be be plotted
    z = float(l[4].split()[2]) # redshift
    ll = str(l[5].split()[2]) #input spectrum (text file)
    fn.close()
    mylog = open('mysplot_log','w')
    #-----------------Reading in the line list & spectrum------------------------------------------------
    list = readlist(ll)
    #inp = '/Users/acharyya/Dropbox/MagE_atlas/Stacked/mage_stacked_spectrum.txt'
    #wz, fl = readmagespec(inp)
    wz, fl = readspec(inp)
    const = np.median(fl)
    if const > 0. and const < 10.:
        const = 1.
    else:
        const = 10**(np.ceil(np.log10(const)))
    wz = obstorest(wz, z)
    fl = shiftflux(fl, const)
    if xmin < 0.:
        xmin = wz[0]
    if xmax < 0.:
        xmax = wz[-1]
    wz, fl = cutspec(wz, fl, xmin, xmax)
    printinstruction(c, inp, xmin, xmax)
    makeplotint(wz, fl, xmin, xmax, c, list, const)
    plt.close()
    mylog.close()
'''
#-----------function for computing continuum------------------------------
def compute_cont():
    cx=[]
    cy=[]
    fn = open('mysplot_log','r')
    l2 = fn.readlines()[1:]
    for l in l2:
        x1 = wz[np.where(wz>float(l.split()[0]))[0][0]:np.where(wz<float(l.split()[1]))[0][len(np.where(wz<float(l.split()[1]))[0])-1]+1]
        p=[float(l.split()[2]),float(l.split()[3]),float(l.split()[4]),float(l.split()[5])]
        cx = np.append(cx, x1)
        x1 = np.array(x1)
        cy = np.append(cy, cont(x1, *p))
    fn.close()
    c = np.array(zip(cx, cy))
    c = c[np.argsort(c[:,0])]
    cnt = interp1d(c[:,0], c[:,1], kind='cubic')
    w = wz[np.where(wz>c[0,0])[0][0]:np.where(wz<c[-1,0])[0][len(np.where(wz<c[-1,0])[0])-1]+1]
    flc = cnt(w)
    ax.plot(w, flc, label='continuum fit', linestyle='--')
    ax.legend()
    plt.draw()
'''