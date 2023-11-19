#!/usr/bin/env bash
#SBATCH --job-name=redis_master                         # name of job
#SBATCH --time=0-4:00:00                                # maximum time until job is cancelled
#SBATCH --cpus-per-task=1                               # number of cpus requested
#SBATCH --mem-per-cpu=4G                                # memory per cpu requested
#SBATCH --mail-type=begin                               # send mail when job begins
#SBATCH --mail-type=end                                 # send mail when job ends
#SBATCH --mail-type=fail                                # send mail if job fails
#SBATCH --mail-user=florian.schunck@ufz.de              # email of user
#SBATCH --output=/work/%u/timepath/logs/%x-%A-%a.out    # output file of stdout messages
#SBATCH --error=/work/%u/timepath/logs/%x-%A-%a.err     # output file of stderr messages

usage() { echo "$0 usage:" && grep " .)\ #" $0; exit 1;}
[ $# -eq 0 ] && usage

while getopts ":c:s:p:a:r:" o; do
    case "${o}" in
        c) # case-study
            CASE=${OPTARG}
            ;;
        s) # case-study
            SCENARIO=${OPTARG}
            ;;
        p) # port
            PORT=${OPTARG}
            ;;
        a) # password
            PASS=${OPTARG}
            ;;
        r) # password
            RUNTIME=${OPTARG}
            ;;
        h | *) # display help
            usage
            exit 1
            ;;
    esac
done





# WARNING: Running the redis master process as a job does not work it has to 
# be launched in an active shell. This is inconvenient but also not a terrible
# problem






module purge

# activate conda environment
module load Anaconda3
source activate timepath-pyabc
module unload Anaconda3

# populate PYTHONPATH (this needs to be done in any new terminal or job, ...)
export PYTHONPATH=~/projects/timepath/case_studies

# this sets up the long term process and waits for workers
tp-abc \
    --case_study $CASE \
    --scenario $SCENARIO \
    --sampler RedisEvalParallelSampler \
    --port $PORT \
    --password $PASS &

echo "running the process for ${RUNTIME} minutes..."

END=$((SECONDS+$RUNTIME*60))

while [[ $SECONDS -lt $END ]]; do
    # get info about workers
    info=`abc-redis-manager info --port $PORT --password $PASS`

    # parse info string
    workers=`echo $info | cut -d " " -f 1`
    evaluations=`echo $info | cut -d " " -f 2`
    accept=`echo $info | cut -d " " -f 3`
    gen=`echo $info | cut -d " " -f 4`

    # parse accept string
    accept_ratio_string=`echo $accept | cut -d "=" -f 2`
    accept_is=`echo $accept_ratio_string | cut -d "/" -f 1`
    accept_target=`echo $accept_ratio_string | cut -d "/" -f 2`
    echo "ABC INFO: runtime=$SECONDS/$END, $info"

    if [[ "$info" = "No active generation" ]]; then
        continue
    else
        accept_ratio=$((accept_is/accept_target))
    fi

    # test if the acceptance ratio is greater equal 1
    if [[ "$accept_ratio" -ge 1 ]]; then
        echo "ABC ACTION: $accept resetting workers to cancel long running processes."
        abc-redis-manager reset-workers --port $PORT --password $PASS
        sleep 30s
    else
        sleep 10s
    fi
done

# cancel workers if script has converged
scancel $wid
redis-cli -p $PORT -a $PASS shutdown