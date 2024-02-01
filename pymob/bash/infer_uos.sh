#!/usr/bin/env bash
#SBATCH --job-name=infer                                # name of job
#SBATCH --time=0-8:00:00                                # maximum time until job is cancelled
#SBATCH --ntasks=1                                      # number of tasks
#SBATCH --cpus-per-task=128                             # number of cpus requested
#SBATCH --mem-per-cpu=4G                                # memory per cpu requested
#SBATCH --mail-type=begin                               # send mail when job begins
#SBATCH --mail-type=end                                 # send mail when job ends
#SBATCH --mail-type=fail                                # send mail if job fails
#SBATCH --mail-user=florian.schunck@uos.de              # email of user
#SBATCH --output=/home/staff/f/fschunck/logs/job-%x-%j.out    # output file of stdout messages
#SBATCH --error=/home/staff/f/fschunck/logs/job-%x-%j.err     # output file of stderr messages


usage() { echo "$0 usage:" && grep " .)\ #" $0; exit 1;}
[ $# -eq 0 ] && usage

while getopts ":c:s:b:" o; do
    case "${o}" in
        c) # case-study
            CASE=${OPTARG}
            ;;
        s) # case-study
            SCENARIO=${OPTARG}
            ;;
        b) # password
            BACKEND=${OPTARG}
            ;;
        h | *) # display help
            usage
            exit 1
            ;;
    esac
done


echo "Requested $SLURM_CPUS_PER_TASK cores."
echo "Starting inference..."
echo "Simulation(case_study=$CASE, scenario=$SCENARIO, backend=$BACKEND)"

# activate conda environment
source activate damage-proxy

# this launches inference
srun pymob-infer \
    --case_study $CASE \
    --scenario $SCENARIO \
    --inference_backend $BACKEND \
    --n_cores $SLURM_CPUS_PER_TASK


echo "Finished inference."
slogs -n $SLURM_JOB_NAME -i $SLURM_JOBID > work/case_studies/$CASE/results/$SCENARIO/log
