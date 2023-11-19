#!/usr/bin/env bash
#SBATCH --job-name=generate_sims
#SBATCH --time=0-12:00:00
#SBATCH --cpus-per-task=1                               # number of cpus requested
#SBATCH --mem-per-cpu=8G
#SBATCH --output=/work/%u/timepath/logs/%x-%A-%a.out
#SBATCH --error=/work/%u/timepath/logs/%x-%A-%a.err
#SBATCH --mail-type=begin                               # send mail when job begins
#SBATCH --mail-type=end                                # send mail when job ends
#SBATCH --mail-type=fail                                # send mail if job fails
#SBATCH --mail-user=florian.schunck@ufz.de              # email of user

CASE_STUDY=$1
SCENARIO=$2
BATCHSIZE=$3

echo "processing chunk $SLURM_ARRAY_TASK_ID ..."
PROJECT_DIR="/home/$USER/projects/timepath"

module purge
module load Anaconda3
source activate timepath-pyabc
module unload Anaconda3 

python3 "$PROJECT_DIR/timepath/generate_sims.py" \
    --case_study $CASE_STUDY \
    --scenario $SCENARIO \
    --worker=$SLURM_ARRAY_TASK_ID \
    --number_simulations=$BATCHSIZE
