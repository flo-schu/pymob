#!/usr/bin/env bash

usage() { echo "$0 usage:" && grep " .)\ #" $0; exit 1;}
[ $# -eq 0 ] && usage

while getopts ":c:s:w:p:a:r:" o; do
    case "${o}" in
        c) # case-study
            CASE=${OPTARG}
            ;;
        s) # case-study
            SCENARIO=${OPTARG}
            ;;
        w) # worker-number
            WORKERS=${OPTARG}
            ;;
        p) # port
            PORT=${OPTARG}
            ;;
        a) # password
            PASS=${OPTARG}
            ;;
        r) # runtime in minutes
            RUNTIME=${OPTARG}
            ;;
        h | *) # display help
            usage
            exit 1
            ;;
    esac
done

# purge all loaded modules
module purge
module load Anaconda3
source activate timepath-pyabc
module unload Anaconda3

RTS=$(($RUNTIME*60))
END=$((SECONDS+RTS))

echo "running case study: ${CASE} with scenario: ${SCENARIO}."
# start server
redis-server config/redis_timepath.conf --port $PORT --requirepass $PASS
echo "launched server on port:${PORT} with password:${PASS}."

# this sets up the long term process and waits for workers
tp-abc \
    --case_study $CASE \
    --scenario $SCENARIO \
    --sampler RedisEvalParallelSampler \
    --port $PORT \
    --password $PASS &

# run loop until runtime of process is reached
echo "running the process for ${RUNTIME} minutes"
sleep 30s

# start workers -t takes runtime in minutes but the workers take runtime in seconds
wid=$(sbatch --parsable -t $RUNTIME -a 1-$WORKERS timepath/bash/abc_redis_worker.sh -p $PORT -a $PASS -r $RTS)
echo "started $WORKERS workers on arry job ${wid}."

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

# sleep a little extra time to allow workers to shutdown
sleep 30

# kill most recent background job (i.e. the master process)
kill %%

# cancel workers if script has converged
scancel $wid
redis-cli -p $PORT -a $PASS shutdown

unset CASE
unset SCENARIO
unset PORT
unset PASS
unset WORKERS
unset RUNTIME
unset wid