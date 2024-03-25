#!/usr/bin/env bash
#SBATCH --job-name=infer                                # name of job
#SBATCH --time=0-1:00:00                                # maximum time until job is cancelled
#SBATCH --ntasks=1                                      # number of tasks
#SBATCH --cpus-per-task=1                               # number of cpus requested
#SBATCH --mem-per-cpu=8G                                # memory per cpu requested
#SBATCH --mail-type=begin                               # send mail when job begins
#SBATCH --mail-type=end                                 # send mail when job ends
#SBATCH --mail-type=fail                                # send mail if job fails
#SBATCH --mail-user=florian.schunck@uos.de              # email of user
#SBATCH --output=/home/staff/f/fschunck/logs/job-%x-%A-%a.out    # output file of stdout messages
#SBATCH --error=/home/staff/f/fschunck/logs/job-%x-%A-%a.err     # output file of stderr messages


usage() { echo "$0 usage:" && grep " .)\ #" $0; exit 1;}
[ $# -eq 0 ] && usage

while getopts ":c:s:b" o; do
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

# specify output directory
PADDED_TASK_ID=$(printf "%05d" $SLURM_ARRAY_TASK_ID)
OUTPUT="/home/staff/f/fschunck/projects/damage-proxy
    work/case_studies/$CASE/results/$SCENARIO/chains/$SLURM_ARRAY_TASK_ID"

echo "Running Array JOB: $PADDED_TASK_ID"
echo "Requested $SLURM_CPUS_PER_TASK cores."
echo "Starting inference..."
echo "Simulation(case_study=$CASE, scenario=$SCENARIO, backend=$BACKEND)"

# activate conda environment
source activate damage-proxy

# this launches inference
srun pymob-infer \
    --case_study $CASE \
    --scenario $SCENARIO \
    --output $OUTPUT \
    --random_seed $SLURM_ARRAY_TASK_ID \
    --n_cores $SLURM_CPUS_PER_TASK \
    --inference_backend $BACKEND 
