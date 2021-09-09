#!/bin/bash -l

#SBATCH -J batch-%j.job
#SBATCH --partition=all
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=30000
#SBATCH --array=163,189,247,300,347,401,453,499
#SBATCH --output=batch-%j.out
#SBATCH --error=batch-%j.err
#SBATCH --time=1-00:00:00


echo "Activating conda environment simulation slices"
conda activate simulation_slices

# limit the number of threads for pygio
export OMP_NUM_THREADS=1
export BATCH_FILE=$1
export N_START=$2
export N_STOP=$3

echo "Running batch job for sim_ids=${N_START}-${N_STOP} from ${BATCH_FILE}"
# run the cli script to generate the full maps for each snapshot for all sims
# snapshots are passed through SBATCH array
# eval srun -n1 --exclusive python map_sims_cli.py $BATCH_FILE ${SLURM_ARRAY_TASK_ID} -i {$N_START..$N_STOP} --save-coords --project-full &
eval srun -n1 --exclusive python map_sims_cli.py $BATCH_FILE ${SLURM_ARRAY_TASK_ID} -i {$N_START..$N_STOP} --save-coords --no-project-full &
# eval srun -n1 --exclusive python map_sims_cli.py $BATCH_FILE ${SLURM_ARRAY_TASK_ID} -i {$N_START..$N_STOP} --no-save-coords --project-full &
wait
