#!/bin/bash
#SBATCH --job-name=UDE_hyperparam
#SBATCH --output=hyperparam_logs/%s_%A_%a.out
#SBATCH --error=hyperparam_logs/%s_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8000MB
#SBATCH --mail-type=END

spack load miniconda3
source activate pymob
spack unload miniconda3

# Get line from input list
line=$(sed -n "${SLURM_ARRAY_TASK_ID}p" input_list.txt)

# Split into variables
IFS=\; read -ra inputs <<<"$line"
length=${inputs[0]}
lr=${inputs[1]}
clip=${inputs[2]}
batch=${inputs[3]}
points=${inputs[4]}
noise=${inputs[5]}

echo "running hyperparameters.py with length=$length, lr=$lr, clip=$clip, batch=$batch, points=$points, and noise=$noise"

# Run simulation
python3 hyperparameters.py -length=$length -lr=$lr -clip=$clip -batch=$batch -points=$points -noise=$noise

echo "running SINDy.py with length=$length, lr=$lr, clip=$clip, batch=$batch, points=$points, and noise=$noise"

# Run simulation
python3 SINDy.py -length=$length -lr=$lr -clip=$clip -batch=$batch -points=$points -noise=$noise

echo "running validation.py with length=$length, lr=$lr, clip=$clip, batch=$batch, points=$points, and noise=$noise"

# Run simulation
python3 validation.py -length=$length -lr=$lr -clip=$clip -batch=$batch -points=$points -noise=$noise