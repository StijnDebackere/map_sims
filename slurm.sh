#!/bin/bash

export BATCH_FILE=$1
export NTASKS=$2

((N_IDS = (111 + $NTASKS - 1) / $NTASKS))

VALS=($(seq 1 $NTASKS))
for i in "${VALS[@]}"
do
    ((N_START=($i - 1) * $N_IDS))
    if [[ $i -eq $NTASKS ]]; then
        ((N_STOP=111))
    else
        ((N_STOP=$i * $N_IDS - 1))
    fi
    sbatch sbatch_single.sh ${BATCH_FILE} ${N_START} ${N_STOP}
    echo "Submitted sbatch run for ${BATCH_FILE} and SIM_IDS ${N_START}-${N_STOP}"
done
