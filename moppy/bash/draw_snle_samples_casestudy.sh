#!/usr/bin/env bash
#SBATCH --job-name=draw_samples                         # name of job
#SBATCH --time=0-8:00:00                                # maximum time until job is cancelled
#SBATCH --cpus-per-task=1                               # number of nodes requested
#SBATCH --mem-per-cpu=4G                                # memory per cpu requested
#SBATCH --output=/work/%u/timepath/logs/%x-%A-%a.out    # output file of stdout messages
#SBATCH --error=/work/%u/timepath/logs/%x-%A-%a.err     # output file of stderr messages
#SBATCH --mail-type=begin                               # send mail when job begins
#SBATCH --mail-type=end                                 # send mail when job ends
#SBATCH --mail-type=fail                                # send mail if job fails
#SBATCH --mail-user=florian.schunck@ufz.de              # email of user

CASE_STUDY=$1
SCENARIO=$2

echo "processing simulations."
PROJECT_DIR="/home/$USER/projects/timepath"

module purge
module load Anaconda3
source activate timepath-pyabc
module unload Anaconda3

python3 "$PROJECT_DIR/timepath/sbi_snle_sample_posterior.py" \
    -c $CASE_STUDY -s $SCENARIO \
    --worker=$SLURM_ARRAY_TASK_ID 