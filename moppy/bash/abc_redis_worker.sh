#!/usr/bin/env bash
#SBATCH --job-name=redis_worker_abc                     # name of job
#SBATCH --time=0-8:10:00                               # maximum time until job is cancelled
#SBATCH --cpus-per-task=1                               # number of cpus requested
#SBATCH --mem-per-cpu=8G                                # memory per cpu requested
#SBATCH --mail-type=begin                               # send mail when job begins
#SBATCH --mail-type=end                                 # send mail when job ends
#SBATCH --mail-type=fail                                # send mail if job fails
#SBATCH --mail-user=florian.schunck@ufz.de              # email of user
#SBATCH --output=/work/%u/timepath/logs/%x-%A-%a.out    # output file of stdout messages
#SBATCH --error=/work/%u/timepath/logs/%x-%A-%a.err     # output file of stderr messages


usage() { echo "$0 usage:" && grep " .)\ #" $0; exit 1;}
[ $# -eq 0 ] && usage

while getopts ":p:a:r:" o; do
    case "${o}" in
        p) # port
            PORT=${OPTARG}
            ;;
        a) # password
            PASS=${OPTARG}
            ;;
        r) # runtime in seconds
            RUNTIME=${OPTARG}
            ;;
        h | *) # display help
            usage
            exit 1
            ;;
    esac
done


echo "setting up redis-worker on array id: ${SLURM_ARRAY_TASK_ID}..."

# prepare environment, e.g. set path
module purge

# activate conda environment
module load Anaconda3
source activate timepath-pyabc
module unload Anaconda3

# populate PYTHONPATH (this needs to be done in any new terminal or job, ...)
export PYTHONPATH=~/projects/timepath/case_studies

# run
abc-redis-worker \
    --host=frontend1 \
    --port=$PORT \
    --password=$PASS \
    --runtime=${RUNTIME}s \
    --catch=1