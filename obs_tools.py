''' package containing usefull routines for obs. data analysis
e.g. lowfreq filtering, stats (pdfs & co), ...
NJAL Dec 2017'''
from __future__ import print_function, division
import numpy as np
from scipy import stats, signal
import obs_tools as otools


def lowfreq_filt(data,fcut,mode='lowpass',use_gust=False,reduce_size=False,verbose=False):
    ''' iterative subsampling with lowpass filtering at each step 
    avoid numerical errors when filtering frequency is low compared to sample frequency
    input fcut is relative cutoff frequency (cycle/time_interval), not Nyquist
    Procedure to reduce the number of points in the output is not optimal
    
    Parameters:
    -----------
    :param 1: data, time series to filter
    :type 1: numpy ndarray (rank 1)
    
    :param 2: fcut: relative cutoff frequency (cycle/time_interval) -> f/fsample
    :type 2: float, scalar or 2-elements ndarray (if mode=='bandpass')
    
    :param 3: mode (optional, default: 'lowpass'): filter mode for last pass
    :type 3: str ('lowpass','bandpass','highpass' (stupid)) -> argument of scipy.signal.filtfilt
    
    :param 4: use_gust (optional, default=True): whether to use gustaffson's method or not at the edges (requires scipy>=0.16.0)
    :type 4: boolean
    
    :param 5: reduce_size (default:False): retain smallest number of points in the output
    :type 5: boolean
    
    :param 6: verbose (optional, default=False): activate verbose mode
    :type 6: boolean
    
    Returned:
    ---------
    :return: data, indices: time series of filtered data and indices of sampling points w.r.t input data
    :rtype: ndarray (rank 1), ndarray (rank 1)
    
    '''
    Nt = len(data)
    if mode=='bandpass':
        freq = np.sqrt(fcut.prod())
    else:
        freq = fcut
    nsub = int(np.log10(1./freq))-1
    # down sampling (with lowpass filter at 1/10)
    if nsub >= 1:
        bb,aa = signal.butter(4,0.1,'lowpass')
        for ii in range(nsub):
            data = signal.filtfilt(bb,aa,data)
            data = data[::10]
        fcut = fcut*10**nsub
    else:
        nsub = 0
    if verbose:
        print('subsampling {0} times before filtering at {1}'.format(nsub,freq*10**nsub))
    bb,aa = signal.butter(4,2.*fcut,mode)
    if use_gust:
        meth = 'gust'
        print('WARNING: using gustafsson but it is not well implemented here... avoid')
    else:
        meth = 'pad'
    data = signal.filtfilt(bb,aa,data,method=meth)
    if reduce_size:
        return data[::2], np.arange(0,Nt,10**nsub*2) # this is clearly not optimal
    else:
        return data, np.arange(0,Nt,10**nsub)

def mystats(data,nbins=None,bnd_bins=None):
    ''' return every statistical information you need in a dictionnary
    
    Parameters:
    -----------
    :input 1: data: time series
    :type 1: ndarray (rank 1)
    
    :input 2: nbins (optional), number of bins. Otherwise choose automatically
    :type 2: int
    
    :input 3: bnd_bins (optional): bounds for bins. Otherwise take min and max
    :type 3: list or tuple of ndarray, 2 floats
    
    Reurned:
    --------
    :return: dictionnary of mean, var, ske, kurt, pdf and bins and binplt
    :rtype: dictionnary
    '''
    
    res = {}
    nobs, minmax, mean, variance, skewness, kurtosis = stats.describe(data)
    if nbins is None:
        nbins = int(np.sqrt(nobs))
    if bnd_bins is None:
        bnd_bins = minmax
    bins = np.linspace(bnd_bins[0],bnd_bins[1],nbins)

    res['mean'] = mean
    res['std'] = np.sqrt(variance)
    #res['skew'] = (skewness, stats.skewtest(data))
    #res['kurt'] = (kurtosis,stats.kurtosistest(data))
    res['skew'] = skewness
    res['kurt'] = kurtosis
    res['skewtest'] = stats.skewtest(data) 
    res['kurtest'] = stats.kurtosistest(data)
    res['bins'] = bins
    res['binplt'] = 0.5*(bins[:-1]+bins[1:])
    res['pdf'] = mypdf(data,bins)
    
    return res

                                     
def mypdf(u,bins):
    ''' compute a normalized pdf '''                                 
    pdf, _ = np.histogram(u,bins)
    pdf = pdf / ((bins[2]-bins[1])*len(u))
    return pdf

def myspectrum(data,dt=1.0,windowing=False):
    ''' compute power spectrum
    windowing is False, can be set to True but no correction in the normalization is applied
    this routine is pretty useless, better use scipy.signal.periodogram routine...'''
    Nt = len(data)
    ff = np.fft.rfftfreq(Nt,dt)
    if windowing:
        window = np.hanning(Nt)
    else:
        window = np.ones(Nt)
    spow = np.abs(np.fft.rfft(data*window))**2*dt/Nt*2
    return spow, ff