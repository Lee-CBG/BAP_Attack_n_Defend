from trl import PPOConfig
from evaluator import gpt_ll, tcr_match


def create_ppo_config(name, configs):
	ppo_config = PPOConfig(
    	model_name= name,
    	learning_rate=configs.learning_rate,  # higher learning rate for better exploration
    	batch_size=configs.batch_size,  # Smaller batch size for more frequent updates
    	mini_batch_size=configs.mini_batch_size,  # 
    	gradient_accumulation_steps=configs.gradient_accumulation_steps,
    	ppo_epochs=configs.ppo_epochs,  # More epochs to better refine the policy in each iteration
    	adap_kl_ctrl=configs.adap_kl_ctrl,  # Disable adaptive KL control
    	init_kl_coef=configs.init_kl_coef,  # Disable the KL divergence penalty
    	cliprange=configs.cliprange,  # increase cliprange to allow larger updates
		)
	return ppo_config

def eval_filter(folder, bap, **kwargs):
	flag = False
	if kwargs.get('gpt_ll'):
		gpt_ll.evaluate(folder, bap)
		flag = True
	if kwargs.get('tcr_match'):
		tcr_match.evaluate(folder, bap)
		flag = True
	return flag