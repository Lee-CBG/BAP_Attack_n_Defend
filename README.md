# BAP_Attack_n_Defend
This is a demo codebase for the BAP attack and defense project with the support of ATM-tcr as the BAP model. For the complete support of variant BAP support and RLHF methods, they will be released with the official codebase after the project is published. 

# Script for reproduction
Please run the bap_attack.py file with the default configuration. It will choose a regression model as the default reward update method (provided by TRL library)
The default BAP model is ATM-TCR for simplicity and computation efficiency.


# data storage
Generated data will be automatically stored at directory **bap_attack/outputs/<Y-M-D>/<H-M-S>/iter_<Number>** folder. The updated BAP model is stored at the same level directory for later evaluation.

## Apply filter 
To generate negative control samples, we apply the filter under **bap_attack/utils/evaluator** directory and it is automatically called in our reproduction script. 

# Envrionment configuration

## bap_attack

Dependencies:
trl==0.11.1 
torch==2.0.1 
pandas==2.2.3
numpy==1.26.4
wandb==0.18.3
hydra-core==1.3.2

## ATM-TCR as BAP
The bap model is a stand-alone module that needs to be forked and configured before running the bap_attack. Here we take ATM-TCR for example.

Step 1: 
fork ATM-tcr from https://github.com/Lee-CBG/ATM-TCR at copy to the same level of bap_attack
--root_dir
  --bap_attack
  --ATM-TCR

Step 2: 
configure conda environment and name as **atm-tcr** for subprocess running.
dependencies:

pytorch==1.5.0
scikit-learn==1.3.2 
numpy==1.24.4
pandas==2.0.3
