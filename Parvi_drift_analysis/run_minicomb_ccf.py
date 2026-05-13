import numpy as np
import matplotlib.pylab as plt
from astropy.io import fits
import glob, sys, argparse, os, warnings
from datetime import datetime,timezone

plt.ion()

from parvi_ccf_tools import *

warnings.filterwarnings('ignore', message='All-NaN slice encountered', category=RuntimeWarning)
warnings.filterwarnings('ignore', message='Mean of empty slice', category=RuntimeWarning)

# Notes
# red/blue doesn't matter bc we use parvi etalon with minicomb
# data from 12/15 wasn't extracted correctly post warmup - not sure why
##############
# RUN SETTINGS
DATETAG_default = '20260512' #datetime.now(timezone.utc).strftime("%Y%m%d")

CONFIGS = {
    'blue': {
        'datapath':   '/Users/ashleybaker/Documents/HISPEC/AIT/CAL_LabData/Etalon_analyses/Parvi_drift_analysis/PARVI_minicomb_data/spectra/',
        'mask_sci':   'parvi_minicomb_ccf_mask_sci.csv',
        'mask_cal':   'parvi_minicomb_ccf_mask_cal.csv',
        'rv_csv':     'running_minicomb_rvs.csv',
        'orders_csv': 'running_minicomb_rvs_byorder.csv',
        'label':      'Minicomb - Parvi Etalon',
        'wavelength_file': '/Users/ashleybaker/Documents/HISPEC/AIT/CAL_LabData/Etalon_analyses/Parvi_drift_analysis/PARVI_redEtalon_data/Altair_R02_20251017031228_deg0_sp.fits',
        'output': 'outputs',
        'orders_to_run': np.arange(10,44)
    },
    'red': {
        'datapath':   '/Users/ashleybaker/Documents/HISPEC/AIT/CAL_LabData/Etalon_analyses/Parvi_drift_analysis/PARVI_minicomb_data/spectra/',
        'mask_sci':   'parvi_minicomb_ccf_mask_sci.csv',
        'mask_cal':   'parvi_minicomb_ccf_mask_cal.csv',
        'rv_csv':     'running_minicomb_rvs.csv',
        'orders_csv': 'running_minicomb_rvs_byorder.csv',
        'label':      'Minicomb - Parvi Etalon',
        'wavelength_file': '/Users/ashleybaker/Documents/HISPEC/AIT/CAL_LabData/Etalon_analyses/Parvi_drift_analysis/PARVI_redEtalon_data/Altair_R02_20251017031228_deg0_sp.fits',
        'output': 'outputs',
        'orders_to_run': np.arange(10,44)
    },
    'minicomb': {

    }
}
##############


def save_rv_results(filename, rvs, rvs_cal, outfile='running_etalon_rvs.csv'):
    """Append (or create) a CSV row: filename, JD, mean RV sci, mean RV cal."""
    hdr = fits.getheader(filename)
    mjd = hdr['TIMEWMJD']

    rv_mean     = np.nanmean(rvs)
    rv_cal_mean = np.nanmean(rvs_cal)

    import csv
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

    import csv
    write_header = not os.path.exists(outfile)
    with open(outfile, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(['filename', 'mjd'] + [f'order_{o}' for o in orders])
        writer.writerow([os.path.basename(filename), mjd] + list(rv_diff))

    return outfile

def plot_summary(all_rv_filename, orders_to_run, label='HISPEC Etalon', color='red', output_dir='.'):
    """Load summary CSV and plot mean RV sci and cal vs time."""
    summary = np.genfromtxt(all_rv_filename, delimiter=',', names=True, dtype=None, encoding=None)

    mjd_all    = np.atleast_1d(summary['mjd']).astype(float)
    rv_sci_all = np.atleast_1d(summary['rv_sci_mean_kms']).astype(float)
    rv_cal_all = np.atleast_1d(summary['rv_cal_mean_kms']).astype(float)
    rv_diff_all = rv_sci_all - rv_cal_all

    hispec_offset = np.nanmean(rv_sci_all)
    parvi_offset  = np.nanmean(rv_cal_all)

    fig, [ax1,ax2] = plt.subplots(2,1 , figsize=(10, 6), num='nightly_summary', sharex=True)
    ax1.plot(mjd_all, 1000*(rv_sci_all - hispec_offset), 'o', c=color, markeredgecolor=color, alpha=0.8, label=label)
    ax1.plot(mjd_all, 1000*(rv_cal_all - parvi_offset),  '^', c='purple',    markeredgecolor='black',     alpha=0.8, label='PARVI Etalon')
    ax1.set_xlabel('MJD [days]')
    ax1.set_ylabel('RV [m/s]')
    ax1.set_title(f'{color} \nOrders {np.min(orders_to_run)} to {np.max(orders_to_run)}')
    ax1.grid()
    ax1.legend()

    # DIFFERENTIAL RV VS TIME
    rv_diff   = 1000 * (rv_diff_all - np.nanmean(rv_diff_all))
    time_avg  = 120
    bin_width = time_avg / 1440
    bin_idx   = np.floor(mjd_all / bin_width).astype(int)
    bin_ids   = np.unique(bin_idx)
    t_binned  = np.array([np.nanmean(mjd_all[bin_idx == b]) for b in bin_ids])
    rv_binned = np.array([np.nanmean(rv_diff[bin_idx == b])    for b in bin_ids])

    ax2.plot(mjd_all, rv_diff, 'ko', alpha=0.2)
    ax2.plot(t_binned, rv_binned, 's', c=color, markeredgecolor='black', ms=6, label=f'{time_avg}-min bin')
    ax2.set_xlabel('MJD [days]')
    ax2.set_ylabel('HISPEC - PARVI [m/s]')
    ax2.legend()
    ax2.grid()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f'running_{color}etalon_rvs.png'))

def plot_orders(all_rvorders_filename, orders_to_run, color='red', output_dir='.'):
    """Load per-order CSV and plot order-to-order, order test, slopes, and differential RV."""
    orders_csv = np.genfromtxt(all_rvorders_filename, delimiter=',', names=True, dtype=None, encoding=None)

    mjd_orders  = np.atleast_1d(orders_csv['mjd']).astype(float)
    rv_diff_all = np.column_stack([np.atleast_1d(orders_csv[f'order_{o}']).astype(float) for o in orders_to_run])

    isort_t = np.argsort(mjd_orders)

    # ORDER TO ORDER
    plt.figure('order_to_order')

    for i, iorder in enumerate(orders_to_run):
        plt.plot(mjd_orders[isort_t], rv_diff_all[:, i][isort_t] - np.nanmean(rv_diff_all[:, i]), '-', alpha=0.7, label=f'order {iorder}')
    plt.xlabel('MJD [days]')
    plt.ylabel('RV diff [km/s]')
    plt.title('HISPEC - PARVI per order (mean-subtracted)')
    plt.grid()
    plt.legend()
    plt.savefig(f'running_{color}etalon_rvs_byorder.png')

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

    plt.figure('order_slopes')
    slope_orders, slopes = zip(*order_slopes)
    plt.scatter(slope_orders, slopes, color=rainbow)
    #plt.plot(np.array(slope_orders), np.array(slopes), color=rainbow)
    plt.xlabel('Order index')
    plt.ylabel('Slope')
    plt.title('Per-order slope of RV residual vs. mean RV')
    plt.grid()
    plt.savefig(os.path.join(output_dir, f'running_{color}etalon_rvs_byorder.png'))


if __name__=='__main__':
    # pick date to run!
    parser = argparse.ArgumentParser()
    parser.add_argument('--datetag', default=DATETAG_default)
    parser.add_argument('--color', default='red', choices=['blue', 'red'])
    args = parser.parse_args()
    DATETAG = args.datetag
    cfg = CONFIGS[args.color]
    os.makedirs(cfg['output'], exist_ok=True)

    if (args.color == 'blue') and DATETAG.startswith('202605'):
        raise ValueError('blue etalon was before 20260501, red etalon was after')
    
    # glob all files and select which ones want to include here
    files = np.sort(glob.glob(cfg['datapath'] + f'*{DATETAG}*fits'))
    nfiles = len(files)
    print(f'Loaded {nfiles} files for date {DATETAG} for the {args.color} etalon')

    # make/LOAD MASK for etalon - parvi etalon file type
    #make_master_mask(files[0:50], save_to_file=True, diagnostics_on=True, file_tag=cfg['mask_sci'].strip('_sci.csv'), wavelength_file=cfg['wavelength_file']) # ONLY MAKE MASTER ONCE
    mask_sci, mask_cal = load_master_mask(sci_file=cfg['mask_sci'], cal_file=cfg['mask_cal'])
    
    # RUN through all files and orders
    parallelize=True # joblib seems to do best!
    if not parallelize:
        rvs = np.zeros((len(files), len(cfg['orders_to_run'])))
        rvs_cal = np.zeros((len(files), len(cfg['orders_to_run'])))
        for i, file in enumerate(files):
            rvs[i], rvs_cal[i] = run_ccf(file, mask_sci, mask_cal, iorder=43,wavelength_file=cfg['wavelength_file'])
    else:
        from joblib import Parallel, delayed
        from tqdm import tqdm
        if nfiles==0: pass
        jobs = [
            (file, iorder)
            for file in files
            for iorder in cfg['orders_to_run']   
        ]

        results = Parallel(n_jobs=10)(
            delayed(run_ccf)(file, mask_sci, mask_cal, iorder=iorder,wavelength_file=cfg['wavelength_file'])
            for file, iorder in tqdm(jobs, desc="CCF", total=len(jobs))
        )

        # Reshape results into (n_iorders, n_files)
        results = np.array(results).reshape( len(files),len(cfg['orders_to_run']), 2)
        rvs     = results[:, :, 0]  # shape: (n_files, n_iorders)
        rvs_cal = results[:, :, 1]

    # SAVE TO FILE (without any offset applied)
    for i, file in enumerate(files):
        all_rv_filename = save_rv_results(file, rvs[i], rvs_cal[i], outfile=os.path.join(cfg['output'], cfg['rv_csv']))
        all_rvorders_filename = save_rv_orders(file, rvs[i], rvs_cal[i], cfg['orders_to_run'], outfile=os.path.join(cfg['output'], cfg['orders_csv']))

    # LOAD FROM FILES AND UPDATE PLOTS
    if nfiles > 1:
        plot_summary(all_rv_filename, cfg['orders_to_run'], label=cfg['label'], color=args.color,output_dir=cfg['output'])
        plot_orders(all_rvorders_filename, cfg['orders_to_run'], color=args.color,output_dir=cfg['output'])

