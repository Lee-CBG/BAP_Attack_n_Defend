# Iterative Attack-and-Defend Framework for Improving TCR-Epitope Binding Prediction Models
This is a demo codebase for the BAP attack and defense project with the support of ATM-tcr as the BAP model. For the complete support of variant BAP support and RLHF methods, they will be released with the official codebase after the project is published. 

## 📜 Abstract

Reliable TCR-epitope binding prediction models are essential for development of adoptive T cell therapy and vaccine design. These models often struggle with false positives, which can be attributed to the limited data coverage in existing negative sample datasets. Common strategies for generating negative samples, such as pairing with background TCRs or shuffling within pairs, fail to account for model-specific vulnerabilities or biologically implausible sequences. To address these challenges, we propose an iterative attack-and-defend framework that systematically identifies and mitigates weaknesses in TCR-epitope prediction models. During the attack phase, a Reinforcement Learning from AI Feedback (RLAIF) framework is used to attack a prediction model by generating biologically implausible sequences that can easily deceive the model. During the defense phase, these identified false positives are incorporated into fine-tuning dataset, enhancing the model's ability to detect false positives. A comprehensive negative control dataset can be obtained by iteratively attacking and defending the model. This dataset can be directly used to improve model robustness, eliminating the need for users to conduct their own attack-and-defend cycles. We apply our framework to five existing binding prediction models, spanning diverse architectures and embedding strategies to show its efficacy. Experimental results show that our approach significantly improves these models' ability to detect adversarial false positives. The combined dataset constructed from these experiments also provides a benchmarking tool to evaluate and refine prediction models. 

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



## 📜 Citation

If you find **TCRGen** useful for your research, please cite our work:

```
@article{
}
```

---

## 🤝 Contributing

We welcome contributions! Please submit a pull request or open an issue if you encounter any problems.
