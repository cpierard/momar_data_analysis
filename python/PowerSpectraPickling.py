from netCDF4 import Dataset
import numpy as np
import scipy.signal as sg
import pickle
import sys

instrument = sys.argv[1] # 'rbr', 'sbe'
years = list(sys.argv[2:])

for year in years:
    print(instrument, year)
    Data = {}
    PowerSpectra = {}

    if instrument == 'rbr':
        dt = 15/(3600*24)
        with Dataset(f'../../netcdf/{year}/rbr.nc', 'r') as nc:
            for i, depth in enumerate(nc['DEPTH'][:]):
                #print(i, depth)
                aux_array = nc['TEMP'][:,i]
                Data[depth] = aux_array[~np.isnan(aux_array)]
                #Data[depth] = np.array(nc['TEMP'][:,i].data)

    elif instrument == 'sbe':
        if year == 2011:
            dt = 180/(3600*24) # 2011 has a period of 180s
        else:
            dt = 360/(3600*24)

        with Dataset(f'../../netcdf/{year}/sbe.nc', 'r') as nc:
            depth_y = list(nc['DEPTH'][:])
            level_label = None

            for d_i in depth_y:
                try:
                    Data[d_i] = np.array(nc[str(d_i)]['TEMP'][:])
                except:
                    print(f'{year} - {d_i} has no TEMP')

    for depth in Data.keys():
        temp = Data[depth]

        freq_aux = np.array([])
        psd_aux = np.array([])

        windows = np.array([49, 35, 10, 2, 0.5])/dt # the size of the windows in days/dt
        lims = [0.08, 1, 3, 10, 20] # limit frequency for windows

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

        PowerSpectra[depth] = {'freq': freq_aux, 'psd': psd_aux}

outfile = open(f'../../support_data/PowerSpectra/psd_{instrument}_{year}','wb')
pickle.dump(PowerSpectra, outfile)
outfile.close()
