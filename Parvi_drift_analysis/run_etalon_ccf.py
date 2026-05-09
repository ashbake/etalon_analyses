import numpy as np
import matplotlib.pylab as plt
from astropy.io import fits
import glob, sys, argparse

plt.ion()

from parvi_ccf_tools import *

# description of parvi data

##############
# RUN SETTINGS
DATETAG_default = '20260423'

CONFIGS = {
    'blue': {
        'datapath':   '/Users/ashleybaker/Documents/HISPEC/AIT/CAL_LabData/EtalonBlue/PARVI_blueEtalon_data/EtalonEtalon/spectra/',
        'mask_sci':   'parvi_blueetalon_ccf_mask_sci.csv',
        'mask_cal':   'parvi_blueetalon_ccf_mask_cal.csv',
        'rv_csv':     'running_blueetalon_rvs.csv',
        'orders_csv': 'running_blueetalon_rvs_byorder.csv',
        'label':      'HISPEC Blue Etalon',
    },
    'red': {
        'datapath':   '/Users/ashleybaker/Documents/HISPEC/AIT/CAL_LabData/EtalonRed/PARVI_redEtalon_data/EtalonEtalon/spectra/',
        'mask_sci':   'parvi_redetalon_ccf_mask_sci.csv',
        'mask_cal':   'parvi_redetalon_ccf_mask_cal.csv',
        'rv_csv':     'running_redetalon_rvs.csv',
        'orders_csv': 'running_redetalon_rvs_byorder.csv',
        'label':      'HISPEC Red Etalon',
    },
}
##############


def save_rv_results(filename, rvs, rvs_cal, outfile='running_etalon_rvs.csv'):
    """Append (or create) a CSV row: filename, JD, mean RV sci, mean RV cal."""
    hdr = fits.getheader(filename)
    mjd = hdr['TIMEWMJD']


    rv_mean     = np.nanmean(rvs)
    rv_cal_mean = np.nanmean(rvs_cal)

    import os, csv
    write_header = not os.path.exists(outfile)
    with open(outfile, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(['filename', 'mjd', 'rv_sci_mean_kms', 'rv_cal_mean_kms'])
        writer.writerow([os.path.basename(filename), mjd, rv_mean, rv_cal_mean])

    return outfile

def save_rv_orders(filename, rvs, rvs_cal, orders, outfile='running_etalon_rvs_byorder.csv'):
    """Append (or create) a CSV row: filename, mjd, then rv_diff per order."""
    hdr = fits.getheader(filename)
    mjd = hdr['TIMEWMJD']

    rv_diff = rvs - rvs_cal  # shape: (n_orders,)

    import os, csv
    write_header = not os.path.exists(outfile)
    with open(outfile, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(['filename', 'mjd'] + [f'order_{o}' for o in orders])
        writer.writerow([os.path.basename(filename), mjd] + list(rv_diff))

    return outfile

def load_and_plot(all_rv_filename, all_rvorders_filename, orders_to_run, label='HISPEC Etalon', tag='etalon'):
    """Load save data and plot all RVs"""
    summary    = np.genfromtxt(all_rv_filename,         delimiter=',', names=True, dtype=None, encoding=None)
    orders_csv = np.genfromtxt(all_rvorders_filename, delimiter=',', names=True, dtype=None, encoding=None)

    mjd_all     = np.atleast_1d(summary['mjd']).astype(float)
    rv_sci_all  = np.atleast_1d(summary['rv_sci_mean_kms']).astype(float)
    rv_cal_all  = np.atleast_1d(summary['rv_cal_mean_kms']).astype(float)
    rv_diff_all = np.column_stack([np.atleast_1d(orders_csv[f'order_{o}']).astype(float) for o in orders_to_run])

    mjd_orders = np.atleast_1d(orders_csv['mjd']).astype(float)

    # ORDER to ORDER
    plt.figure('order_to_order')
    # sort mjd_orders
    isort_t = np.argsort(mjd_orders)

    for i, iorder in enumerate(orders_to_run):
        plt.plot(mjd_orders[isort_t], rv_diff_all[:, i][isort_t] - np.nanmean(rv_diff_all[:, i]), '-', alpha=0.7, label=f'order {iorder}')
    plt.xlabel('MJD [days]')
    plt.ylabel('RV diff [km/s]')
    plt.title('HISPEC - PARVI per order (mean-subtracted)')
    plt.grid()
    plt.legend()
    plt.savefig(f'running_{tag}etalon_rvs_byorder.png')

    # ORDER TEST
    plt.figure()
    rainbow = plt.cm.rainbow(np.linspace(0, 1, len(orders_to_run)))
    order_slopes = []
    for i, iorder in enumerate(orders_to_run):
        # bin in time here
        rv_diff_order   = 1000 * (rv_diff_all[:, i][isort_t] - np.nanmean(rv_diff_all[:, i]))
        rv_diff = np.nanmean(rv_diff_all,axis=1)[isort_t]
        bin_width = 160 / 1440  # 30 minutes in days
        bin_idx   = np.floor(mjd_orders / bin_width).astype(int)
        bin_ids   = np.unique(bin_idx)
        t_binned  = np.array([np.nanmean(mjd_orders[bin_idx == b])   for b in bin_ids])
        rv_binned = np.array([np.nanmean(rv_diff[bin_idx == b]) for b in bin_ids])
        rv_order_binned = np.array([np.nanmean(rv_diff_order[bin_idx == b]) for b in bin_ids])

        plt.scatter(rv_binned, rv_order_binned, color=rainbow[i], label=f'order {iorder}')
        mask = np.isfinite(rv_binned) & np.isfinite(rv_order_binned)
        if mask.sum() >= 2:
            coeffs = np.polyfit(rv_binned[mask], rv_order_binned[mask], 1)
            x_fit = np.linspace(rv_binned[mask].min(), rv_binned[mask].max(), 100)
            plt.plot(x_fit, np.polyval(coeffs, x_fit), color=rainbow[i])
            order_slopes.append((iorder, coeffs[0]))
        else:
            order_slopes.append((iorder, np.nan))

    plt.figure()
    slope_orders, slopes = zip(*order_slopes)
    plt.scatter(slope_orders, slopes, color=rainbow)
    plt.plot(np.array(slope_orders), np.array(slopes), color=rainbow)
    plt.xlabel('Order index')
    plt.ylabel('Slope')
    plt.title('Per-order slope of RV residual vs. mean RV')
    plt.grid()
    

    # DEFINE OFFSETS for plotting since we'll make night to night plots
    hispec_offset = np.nanmean(rv_sci_all)
    parvi_offset = np.nanmean(rv_cal_all)

    # NIGHT TO NIGHT PLOT
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, num='nightly')
    ax1.plot(mjd_all, 1000*(rv_sci_all - hispec_offset), 'o', c='steelblue', markeredgecolor='steelblue',  alpha=0.8, label=label)
    ax1.plot(mjd_all, 1000*(rv_cal_all - parvi_offset),  '^', c='purple', markeredgecolor='black', alpha=0.8, label='PARVI Etalon')
    ax1.set_ylabel('RV [m/s]')
    ax1.set_title(f'Orders {np.min(orders_to_run)} to {np.max(orders_to_run)}')
    ax1.grid()
    ax1.legend()

    #rv_diff   = 1000 * (rv_sci_all - rv_cal_all - (hispec_offset - parvi_offset))
    rv_diff = 1000 * (np.nanmean(rv_diff_all[:, 8:-2], axis=1) - np.nanmean(rv_diff_all[:, 8:-2]))
    time_avg = 120
    bin_width = time_avg / 1440  # 30 minutes in days
    bin_idx   = np.floor(mjd_orders / bin_width).astype(int)
    bin_ids   = np.unique(bin_idx)
    t_binned  = np.array([np.nanmean(mjd_orders[bin_idx == b]) for b in bin_ids])
    rv_binned = np.array([np.nanmean(rv_diff[bin_idx == b]) for b in bin_ids])

    ax2.plot(mjd_orders, rv_diff, 'ko', alpha=0.2)
    ax2.plot(t_binned, rv_binned, 's',c='steelblue', markeredgecolor='black', ms=6, label=f'{time_avg}-min bin')
    ax2.set_xlabel('MJD [days]')
    ax2.set_ylabel('HISPEC - PARVI [m/s]')
    ax2.legend()
    ax2.grid()

    fig.tight_layout()
    fig.savefig(f'running_{tag}etalon_rvs.png')

if __name__=='__main__':
    # pick date to run!
    parser = argparse.ArgumentParser()
    parser.add_argument('--datetag', default=DATETAG_default)
    parser.add_argument('--color', default='blue', choices=['blue', 'red'])
    args = parser.parse_args()
    DATETAG = args.datetag
    cfg = CONFIGS[args.color]

    # glob all files and select which ones want to include here
    files = np.sort(glob.glob(cfg['datapath'] + f'*{DATETAG}*fits'))
    nfiles = len(files)
    print(f'Loaded {nfiles} files')

    # make/load mask for etalon - parvi etalon file type
    #make_master_mask(files[0:50], save_to_file=True, diagnostics_on=True) # ONLY MAKE MASTER ONCE
    mask_sci, mask_cal = load_master_mask(sci_file=cfg['mask_sci'], cal_file=cfg['mask_cal'])
    
    # initiate arrays
    orders_to_run = np.arange(4,40)
    rvs = np.zeros((len(files), len(orders_to_run)))
    rvs_cal = np.zeros((len(files), len(orders_to_run)))

    # run through all files and orders
    parallelize='joblib' # joblib seems to do best!
    if parallelize == 'no':
        for i, file in enumerate(files):
            rvs[i], rvs_cal[i] = run_ccf(file, mask_sci, mask_cal, iorder=43)
    elif parallelize == 'pool':
        from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
        for ifile, file in enumerate(files):
            print(ifile)
            # Use ProcessPoolExecutor instead if run_ccf is CPU-bound (pure Python/numpy)
            #with ThreadPoolExecutor() as executor:
            with ProcessPoolExecutor() as executor:
                futures = {executor.submit(run_ccf, file, mask_sci, mask_cal, iorder=iorder): i 
                        for i, iorder in enumerate(orders_to_run)}
                
                for future in as_completed(futures):
                    i = futures[future]
                    rvs[ifile,i], rvs_cal[ifile,i] = future.result()
    elif parallelize =='joblib':
        from joblib import Parallel, delayed
        from tqdm import tqdm

        jobs = [
            (file, iorder)
            for file in files
            for iorder in orders_to_run   
        ]

        results = Parallel(n_jobs=10)(
            delayed(run_ccf)(file, mask_sci, mask_cal, iorder=iorder)
            for file, iorder in tqdm(jobs, desc="CCF", total=len(jobs))
        )

        # Reshape results into (n_iorders, n_files)
        results = np.array(results).reshape( len(files),len(orders_to_run), 2)
        rvs     = results[:, :, 0]  # shape: (n_files, n_iorders)
        rvs_cal = results[:, :, 1]

    # SAVE TO FILE (without any offset applied)
    for i, file in enumerate(files):
        all_rv_filename = save_rv_results(file, rvs[i], rvs_cal[i], outfile=cfg['rv_csv'])
        all_rvorders_filename = save_rv_orders(file, rvs[i], rvs_cal[i], orders_to_run, outfile=cfg['orders_csv'])


    # LOAD FROM FILES AND UPDATE PLOTS
    if len(files) > 1:
        load_and_plot(all_rv_filename, all_rvorders_filename, orders_to_run,
                      label=cfg['label'], tag=args.color)

