#!/usr/bin/env bash
#SBATCH --job-name=eval_sbi                             # name of job
#SBATCH --time=0-01:00:00                               # maximum time until job is cancelled
#SBATCH --mem-per-cpu=4G                                # memory per cpu requested
#SBATCH --output=/work/%u/timepath/logs/%x-%A-%a.out    # output file of stdout messages
#SBATCH --error=/work/%u/timepath/logs/%x-%A-%a.err     # output file of stderr messages
#SBATCH --mail-type=begin                               # send mail when job begins
#SBATCH --mail-type=end                                 # send mail when job ends
#SBATCH --mail-type=fail                                # send mail if job fails
#SBATCH --mail-user=florian.schunck@ufz.de              # email of user

CASE_STUDY=$1
SCENARIO=$2

echo "evaluating MCMC chains."
PROJECT_DIR="/home/$USER/projects/timepath"

module purge
module load Anaconda3
source activate timepath-pyabc
module unload Anaconda3 

python3 "$PROJECT_DIR/timepath/evaluate_sbi.py" \
    -c $CASE_STUDY -s $SCENARIO 
