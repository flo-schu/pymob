CASE_STUDY=$1        # directory containing the case study
SCENARIO=$2          # directory in $CASE_STUDY/scenarios that containes the scenario
SIM_ARRAYS=$3        # 1-100 (e.g.) arrays for simulation job
SIM_BATCHSIZE=$4     # 1000 number of simulations per job
MCMC_ARRAYS=$5       # 1-10 arrays for mcmc sampling job
POSTERIOR_BATCHSIZE="10"
POSTERIOR_ARRAYS="1-100"

SCRIPT1="timepath/bash/generate_simulations_casestudy.sh"
jobid_1=$(sbatch -a $SIM_ARRAYS --parsable $SCRIPT1 $CASE_STUDY $SCENARIO $SIM_BATCHSIZE)
echo "submitted job $jobid_1"

SCRIPT2="timepath/bash/process_simulations_casestudy.sh"
jobid_2=$(sbatch --parsable --dependency=afterany:${jobid_1} $SCRIPT2 $CASE_STUDY $SCENARIO)
echo "submitted job $jobid_2"

# SCRIPT2a="timepath/bash/prior_predictive_checks_casestudy.sh"
# jobid_2a=$(sbatch --parsable --dependency=afterany:${jobid_2} $SCRIPT2a $CASE_STUDY $SCENARIO)
# echo "submitted job $jobid_2a"

SCRIPT3="timepath/bash/train_network_casestudy.sh"
jobid_3=$(sbatch --parsable --dependency=afterany:${jobid_2} $SCRIPT3 $CASE_STUDY $SCENARIO "SNPE")
echo "submitted job $jobid_3"

jobid_3b=$(sbatch --parsable --dependency=afterany:${jobid_2} $SCRIPT3 $CASE_STUDY $SCENARIO "SNLE")
echo "submitted job $jobid_3b"

SCRIPT4="timepath/bash/draw_snle_samples_casestudy.sh"
jobid_4=$(sbatch -a $MCMC_ARRAYS --parsable --dependency=afterany:${jobid_3b} $SCRIPT4 $CASE_STUDY $SCENARIO)
echo "submitted job $jobid_4"

SCRIPT5="timepath/bash/evaluate_sbi_casestudy.sh"
jobid_5=$(sbatch --parsable --dependency=afterany:${jobid_4} $SCRIPT5 $CASE_STUDY $SCENARIO)
echo "submitted job $jobid_5"

SCRIPT6="timepath/bash/posterior_predictions_casestudy.sh"
jobid_6=$(sbatch -a $POSTERIOR_ARRAYS --parsable --dependency=afterany:${jobid_5} $SCRIPT6 $CASE_STUDY $SCENARIO $POSTERIOR_BATCHSIZE)
echo "submitted job $jobid_6"

SCRIPT7="timepath/bash/plot_posterior_predictions_casestudy.sh"
jobid_7=$(sbatch --parsable --dependency=afterany:${jobid_6} $SCRIPT7 $CASE_STUDY $SCENARIO)
echo "submitted job $jobid_7"
