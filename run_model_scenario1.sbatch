#!/bin/bash
#SBATCH --job-name=array_zip
#SBTACH --ntasks=1 
#SBATCH --array=1-14
#SBATCH --time=30:00:00 
#SBATCH -p serc 
#SBATCH --mem-per-cpu=8GB 
#SBATCH --cpus-per-task=8 
python3 integrated_model_run.py -p $SLURM_ARRAY_TASK_ID -n sixyears_scenario1_individual 