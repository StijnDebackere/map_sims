#!/bin/bash -l

# --- CALLED FROM map_sims_slurm_run.py ---
# Batch call to map_sims_cli.py for a subset N_START-N_STOP of the
# simulations in BATCH_FILE
echo "Activating conda environment simulation slices"
conda activate simulation_slices

# limit the number of threads for pygio
export OMP_NUM_THREADS=1
export BATCH_FILE=$1
export N_START=$2
export N_STOP=$3

# determine which combination of --save-info and --project-full to run
# save-info | project-full | FLAG
#     0     |      0       |   0
#     1     |      0       |   1
#     0     |      1       |   2
#     1     |      1       |   3
export FLAG=$4

# run the cli script to generate the full maps for each snapshot for all sims
# snapshots are passed through SBATCH array
echo "Running batch job for sim_ids=${N_START}-${N_STOP} from ${BATCH_FILE} for snapshot=${SLURM_ARRAY_TASK_ID}"
if [[ $FLAG -eq 1 ]]; then
    eval srun -n1 --exclusive map_sims ${BATCH_FILE} -s ${SLURM_ARRAY_TASK_ID} -i {$N_START..$N_STOP} --save-info --no-project-full &
elif [[ $FLAG -eq 2 ]]; then
    eval srun -n1 --exclusive map_sims ${BATCH_FILE} -s ${SLURM_ARRAY_TASK_ID} -i {$N_START..$N_STOP} --no-save-info --project-full &
elif [[ $FLAG -eq 3 ]]; then
    eval srun -n1 --exclusive map_sims ${BATCH_FILE} -s ${SLURM_ARRAY_TASK_ID} -i {$N_START..$N_STOP} --save-info --project-full &
else
    echo "Doing nothing"
fi
wait
