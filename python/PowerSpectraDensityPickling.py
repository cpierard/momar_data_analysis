from netCDF4 import Dataset
import numpy as np
import scipy.signal as sg
import pickle
import sys
import gsw

# instrument = sys.argv[1] # 'rbr', 'sbe'
years = list(sys.argv[1:])

for year in years:
    print(year)
    Data = {}
    PowerSpectra = {}

    if year == 2011:
        dt = 180/(3600*24) # 2011 has a period of 180s
    else:
        dt = 360/(3600*24)

    with Dataset(f'../../netcdf/{year}/sbe.nc', 'r') as nc:

        depth_y = list(nc['DEPTH'][:])
        #print(depth_y)

        for d_i in depth_y:

            try:
                temp_aux = np.array(nc[str(d_i)]['TEMP'][:])
                sal_aux = np.array(nc[str(d_i)]['PSAL'][:])
                pres_aux = np.array(nc[str(d_i)]['PRES'][:])
                Data[d_i] = gsw.density.rho_t_exact(sal_aux, temp_aux, pres_aux)

            except:
                print(f'Not able to compute density for {year} - {d_i}.')

    for depth in Data.keys():
        # computing density
        density = Data[depth]

        # Power spectra analysis
        freq_aux = np.array([])
        psd_aux = np.array([])

        windows = np.array([49, 35, 10, 2, 0.5])/dt # the size of the windows in days/dt
        lims = [0.08, 1, 3, 10, 20] # limit frequency for windows

        freq, psd = sg.periodogram(density, fs=1./dt)
        ind, = np.where(freq < lims[0])
        freq_aux = np.concatenate((freq_aux, freq[ind]))
        psd_aux = np.concatenate((psd_aux, psd[ind]))

        for i in range(0,len(lims)-1):
            freq, psd = sg.welch(density, fs=1./dt, window='hanning', nperseg=windows[i], noverlap=0.15)
            ind, = np.where(((freq >= lims[i]) & (freq < lims[i+1])))
            freq_aux = np.concatenate((freq_aux, freq[ind]))
            psd_aux = np.concatenate((psd_aux, psd[ind]))

        freq, psd = sg.welch(density, fs=1./dt, window='hanning', nperseg=windows[-1], noverlap=0.15)
        ind, = np.where(freq >= lims[-1])
        freq_aux = np.concatenate((freq_aux, freq[ind]))
        psd_aux = np.concatenate((psd_aux, psd[ind]))

        PowerSpectra[depth] = {'freq': freq_aux, 'psd': psd_aux}

    outfile = open(f'../../support_data/PowerSpectra/psd_density_{year}','wb')
    pickle.dump(PowerSpectra, outfile)
    outfile.close()
    print('------------------')
