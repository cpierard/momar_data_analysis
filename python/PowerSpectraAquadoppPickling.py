from netCDF4 import Dataset
import numpy as np
import scipy.signal as sg
import pickle
import sys

instrument = 'aquadopp' # 'rbr', 'sbe'
years = list(sys.argv[1:])

for year in years:

    if year == 2013:
        continue

    print(instrument, year)
    Data = {}
    PowerSpectra = {}

    if year in ['2011']:
        dt = 180/(3600*24) # 2011 has a period of 180s
    elif year in ['2012']:
        dt = 6/(60*24)
    elif year in ['2014', '2015', '2018', '2019']:
        dt = 12/(60*24) # 2014 has a period of 12min
    elif year in ['2016', '2017']:
        dt = 10/(60*24) # 10 min

    with Dataset(f'../../netcdf/{year}/aquadopp.nc', 'r') as nc:
        Data['speed'] = np.array(nc['HCSP'][:])
        Data['dir'] = np.array(nc['HCDT'][:])*np.pi/180 # change to radians

    for key in Data.keys():
        #print(key, Data[key].mean())

        temp = Data[key]

        freq_aux = np.array([])
        psd_aux = np.array([])

        windows = np.array([100, 20, 10, 2, 0.5])/dt # the size of the windows in days/dt
        lims = [0.08, 1.2, 3, 10, 20] # limit frequency for windows

        freq, psd = sg.periodogram(temp, fs=1./dt)
        ind, = np.where(freq < lims[0])
        freq_aux = np.concatenate((freq_aux, freq[ind]))
        psd_aux = np.concatenate((psd_aux, psd[ind]))

        for i in range(0,len(lims)-1):
            freq, psd = sg.welch(temp, fs=1./dt, window='hanning', nperseg=windows[i], noverlap=0.15)
            ind, = np.where(((freq >= lims[i]) & (freq < lims[i+1])))
            freq_aux = np.concatenate((freq_aux, freq[ind]))
            psd_aux = np.concatenate((psd_aux, psd[ind]))

        freq, psd = sg.welch(temp, fs=1./dt, window='hanning', nperseg=windows[-1], noverlap=0.15)
        ind, = np.where(freq >= lims[-1])
        freq_aux = np.concatenate((freq_aux, freq[ind]))
        psd_aux = np.concatenate((psd_aux, psd[ind]))

        PowerSpectra[key] = {'freq': freq_aux, 'psd': psd_aux}

    outfile = open(f'../../support_data/PowerSpectra/psd_{instrument}_{year}','wb')
    pickle.dump(PowerSpectra, outfile)
    outfile.close()
