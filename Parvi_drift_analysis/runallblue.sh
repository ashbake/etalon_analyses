#!/bin/bash
#SBATCH --job-name=blueetalon  # Name of the job
#SBATCH --partition=normal     # Target queue/partition
#SBATCH --ntasks=1             # Number of tasks/cores
#SBATCH --mem=10G              # Memory required (e.g., 2GB)
#SBATCH --time=02:00:00        # Maximum runtime (HH:MM:SS)
#SBATCH --array=0-4            # indices into the dates array

for i in 20251127 20251128 20251129 20251130 20251131 20251201 20251202; do
    python run_etalon_ccf.py --datetag $i
done


RUN DATES -these are donezo
--------
20260423
20260422
20260412
20260401 - 0 files
20260421
20260409
20260408
20260410
20260411
20260415
20260413 20260414 20260416 20260417 20260418 20260419 20260420 20260407 [done - 80min]
20260406 20260405 20260403 20260402 20251114 20251115 20251116 [done - 80min]
20251117 20251118 20251119 20251120 20251121 20251122 20251123 20251124 20251125 20251126 20251127 [done - 80min]
20251127 20251128 20251129 20251130 20251131 20251201 20251202 [running - 80min]

