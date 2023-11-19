#!/usr/bin/env bash
#SBATCH --job-name=test_sbi_pipeline_casestudy          # name of job
#SBATCH --time=0-2:00:00                                # maximum time until job is cancelled
#SBATCH --cpus-per-task=1                               # number of cpus requested
#SBATCH --mem-per-cpu=4G                                # memory per cpu requested
#SBATCH --output=/work/%u/timepath/logs/%x-%A-%a.out    # output file of stdout messages
#SBATCH --error=/work/%u/timepath/logs/%x-%A-%a.err     # output file of stderr messages
#SBATCH --mail-type=begin                               # send mail when job begins
#SBATCH --mail-type=end                                 # send mail when job ends
#SBATCH --mail-type=fail                                # send mail if job fails
#SBATCH --mail-user=florian.schunck@ufz.de              # email of user

echo "calling draw samples program ..."

module purge
module load Anaconda3
source activate timepath-pyabc
module unload Anaconda3 

pytest tests/test_scripts.py::TestSbiPipeline