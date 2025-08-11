#!/bin/bash
# FILENAME:  degen

#SBATCH --job-name=degen_comps
#SBATCH --partition=comp
#SBATCH --exclusive
#SBATCH --time=5:00:00
#SBATCH --output=/data101/makinen/joblogs/degen_job_%A_%a.out    # Std-out (%A jobID, %a taskID)
#SBATCH --error=/data101/makinen/joblogs/degen_job_%A_%a.err     # Std-err
#SBATCH --mail-type=all       # Send email to above address at begin and end of job
#SBATCH --mem=2G
#SBATCH --array=0-6                    # <-- run tasks 0 … 9 (10 jobs)
 
# all of the other things here


module load cuda/12.3
module load openmpi

XLA_FLAGS=--xla_gpu_cuda_data_dir=\${CUDA_PATH}
export XLA_FLAGS

source /home/makinen/mysofts/anaconda3/bin/activate

conda init --all
conda activate pyoperon

# cd to top-level directory
cd /home/makinen/repositories/degen_discovery/data/




# --‐ Optional: load modules or activate conda/venv ----------------
# module load python/3.10
# conda activate my_env

# --‐ Run the Python script, forwarding the array index ------------
# python my_script.py --task-id "${SLURM_ARRAY_TASK_ID}"


# python test_sr_full.py --filename fake_cmb_flatten_sqrt.npz --datapath ./degen_data/ --component-id 2 --sr-time 32 --sr-folder symbolic_components_F_Fstd --rotation-type Fstar_Fstd


# test each component, then test rotation methods

python test_sr_full.py --component-id "${SLURM_ARRAY_TASK_ID}" --filename fake_cmb_flatten_sqrt.npz --datapath ./degen_data/ --sr-time 32 --sr-folder symbolic_components_F_Fstd --rotation-type Fstar_Fstd