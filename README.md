# BAP_Attack_n_Defend
This is a demo codebase for the BAP attack and defense project with the support of ATM-tcr as the BAP model. For the complete support of variant BAP support and RLHF methods, they will be released with the official codebase after the project is published. 

# Script for reproduction
Please run the bap_attack.py file with the default configuration. It will choose a regression model as the default reward update method (provided by TRL library)
The default BAP model is ATM-TCR for simplicity and computation efficiency.


# data storage
Generated data will be automatically stored at directory **bap_attack/outputs/<Y-M-D>/<H-M-S>/iter_<Number>** folder. The updated BAP model is stored at the same level directory for later evaluation.

## Apply filter 
To generate negative control samples, we apply the filter under **bap_attack/utils/evaluator** directory and it is automatically called in our reproduction script. 

