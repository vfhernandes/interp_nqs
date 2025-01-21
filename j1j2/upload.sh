#!/bin/sh

for i in $(seq 648 863)
do
    sbatch run.sh $i
done
