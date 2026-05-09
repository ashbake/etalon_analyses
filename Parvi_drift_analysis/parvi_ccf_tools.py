import numpy as np
import matplotlib.pylab as plt
from astropy.io import fits
import glob, sys

import ccf


def load_parvi_data(filename):
    """
    load filename of parvi spectral file

    0-  f[2].data[0,norder,:] # wavelength Wavelength - andrea inserts the wavelength solution here, right now it is hybrid UNe + LFC wavesol
    1-  Extracted Flux
    2-  Measurement of the error in the extracted flux [1./sqrt(Var)]
    3-  Blaze corrected Flux
    4-  Measurement of the error in the blaze corrected flux
    5-  Corresponding fiber flat extraction
    6-  Error of corresponding fiber flat
    etalon is in extension 4, science in extension 2
    """
    # load single fits data
    f = fits.open(filename)
    science_extension = 2 # ch3?
    cal_extension = 4 # ch1?

    wave_sci = f[science_extension].data[0, :, :]
    flux_sci = f[science_extension].data[1, :, :]
    err_sci =  f[science_extension].data[2, :, :]

    wave_cal = f[cal_extension].data[0, :, :]
    flux_cal = f[cal_extension].data[1, :, :]
    err_cal =  f[cal_extension].data[2, :, :]

    return [wave_sci, flux_sci, err_sci], [wave_cal, flux_cal, err_cal]

def load_wavelength_array(filename = '/Altair_R02_20251017031228_deg0_sp.fits'):
    """loads old altair file to get old wavelength solution which shouldn't need to be accurate here
    because parvi data doesn't always ahve wavelengths stored (TODO to fix this)"""
    #filename = '/Users/ashleybaker/Documents/DLC/_data/Altair/Altair_R02_20251017031228_deg0_sp.fits'
    test_sci, test_cal = load_parvi_data(filename)

    waves_sci = test_sci[0]
    waves_cal = test_cal[0]

    return waves_sci, waves_cal

def make_master_mask(files, save_to_file=True, diagnostics_on=False, file_tag='parvi_blueetalon_ccf_mask'):
    """ Make mask for etalon for each order and optionally save to file

    input:
    -----
    files - np.array
        list of files to include in mask
    
    """
    sci, cal = load_parvi_data(files[0])

    # Make master file
    sci_master, cal_master = np.zeros_like(sci[1]), np.zeros_like(cal[1])
    for file in files:
        sci, cal = load_parvi_data(file)
        sci_master += sci[1]
        cal_master += cal[1]

    sci_master/= len(files)
    cal_master/= len(files)

    nfiles = len(files)
    norders, npix = np.shape(sci_master)

    # load wavelength
    w_sci, w_cal = load_wavelength_array()
    
    # Peak finder time
    from scipy.signal import find_peaks
    
    if save_to_file:
        cal_file = file_tag + '_cal.csv'
        sci_file = file_tag + '_sci.csv'
    else:
        cal_file, sci_file = None, None

    if save_to_file:  fcal = open(cal_file, 'w')
    if save_to_file:  fsci = open(sci_file, 'w')
    all_cens = {}
    all_weights={}
    all_cens_cal = {}
    all_weights_cal ={}
    norm_factor=100000 # so weights are between 0 and 1 for easier reading
    for iorder in np.arange(norders):
        # these peaking finding params were optimized
        ipeaks, _ = find_peaks(sci_master[iorder], prominence=0.5*np.nanmedian(sci_master[iorder]), width=3, distance=10)
        ipeaks_cal, _ = find_peaks(cal_master[iorder], prominence=0.5*np.nanmedian(cal_master[iorder]), width=3, distance=10)

        # save to dictionaries for each iorder
        all_cens[iorder]    = w_sci[iorder][ipeaks]
        all_weights[iorder] = np.sqrt(sci_master[iorder][ipeaks] / norm_factor) # sqrt to go to SNR scale
        all_cens_cal[iorder]    = w_cal[iorder][ipeaks_cal]
        all_weights_cal[iorder] = np.sqrt(cal_master[iorder][ipeaks_cal] / norm_factor) # sqrt to go to SNR scale

        # save to file
        if save_to_file:
            for i in np.arange(len(ipeaks)):
                fsci.write(f'{iorder},{all_cens[iorder][i]},{all_weights[iorder][i]}\n')
            for i in np.arange(len(ipeaks_cal)):
                fcal.write(f'{iorder},{all_cens_cal[iorder][i]},{all_weights_cal[iorder][i]}\n')

    # plot if need to diagnose peak finding issue
    if diagnostics_on:
        plt.figure()
        plt.plot(w_sci[iorder], sci_master[iorder])
        plt.plot(w_sci[iorder][ipeaks], sci_master[iorder][ipeaks],'o')
        plt.plot(w_cal[iorder], cal_master[iorder])
        plt.plot(w_cal[iorder][ipeaks_cal], cal_master[iorder][ipeaks_cal],'o')
        plt.title(f'Order {iorder}')

    if save_to_file: fsci.close(), fcal.close()

    return all_cens, all_weights, all_cens_cal, all_weights_cal, cal_file, sci_file

def load_master_mask(sci_file='parvi_blueetalon_ccf_mask_sci.csv', cal_file='parvi_blueetalon_ccf_mask_cal.csv'):
    """loads mask files (hardcoded filenames)
    
    returns 
    --------
    pandas array of sci and cal masks, respectively
        Each pd array has 'order', 'cen', 'weight' columns
     """
    import pandas as pd

    mask_sci = pd.read_csv(sci_file, names=['order','cen','weight'])
    mask_cal = pd.read_csv(cal_file, names=['order','cen','weight'])

    return mask_sci, mask_cal

def run_ccf(filename, mask_sci, mask_cal, iorder=10, wavelength_file='/Altair_R02_20251017031228_deg0_sp.fits'):
    # load file and wave array
    sci, cal = load_parvi_data(filename) # each is indexed wave, flux, err
    w_sci, w_cal = load_wavelength_array(wavelength_file)

    # define mask parameters
    R = 700000
    nsamp=3
    SPEEDOFLIGHT = 299792.458 # km/s
    mask_wid  = (SPEEDOFLIGHT / R) / nsamp #velocity_per_pix
    velocity_loop = np.arange(-20, 20, mask_wid/4)

    # select order from the mask
    mask_sci_order = mask_sci[mask_sci['order'] == iorder]
    mask_cal_order = mask_cal[mask_cal['order'] == iorder]

    # create ccf for that order
    if len(mask_sci_order['cen'].values) > 2:
        try:
            ccf_out_sci, _ = ccf.ccf_make(w_sci[iorder], sci[1][iorder], mask_sci_order['cen'].values, mask_wid*10,  mask_sci_order['weight'].values, velocity_loop, R=R,nsamp=nsamp)
            ccf_out_cal, _ = ccf.ccf_make(w_cal[iorder], cal[1][iorder],  mask_cal_order['cen'].values, mask_wid*10,  mask_cal_order['weight'].values, velocity_loop, R=R,nsamp=nsamp)

            # fit ccf for rv
            rv_fit_sci, v_fit, y_fit, xi2 = ccf.fit_gaussian_to_ccf_2(velocity_loop, 1 - ccf_out_sci/np.max(ccf_out_sci), 0.1, velocity_halfrange_to_fit=5.0,stddev_guess=1)
            rv_fit_cal, _, _, _ = ccf.fit_gaussian_to_ccf_2(velocity_loop, 1 - ccf_out_cal/np.max(ccf_out_cal), 0.1, velocity_halfrange_to_fit=5.0,stddev_guess=1)
        except ValueError:
            return np.nan, np.nan
    else:
        return np.nan, np.nan
    
    return rv_fit_sci, rv_fit_cal

