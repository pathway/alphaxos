env_label='c4'
dqn_label='c4_dqn3'

regime_params_base = {
	'env': env_label, 'dqn_kind': dqn_label,
	'gamma': 0.99,
	'epsilon-train': 0.5,
	'epsilon-chaos': 0.5,
	'delta_window':0.1,
	'learning_rate': 1e-3,
	'memory_batch_size': 128, 'steps_warmup': 100, 'steps_target_model_update': 100, 'steps_per_iter': 300,
	'test_rounds': 50,
	'memory_iterations_per_save': 3,
	'reward_invalid_move': -2.0
}


regime_params_wakeup = {
	'learning_rate': 1e-2,
}



regime_params_early = {
	'epsilon-train':0.5,
	'epsilon-chaos':0.5,
	'learning_rate': 1e-4,
	'test_rounds': 100,

}

regime_params_late = {
	'learning_rate': 1e-5,
	#'memory_batch_size': 128,'steps_warmup': 100,'steps_target_model_update':100,'steps_per_iter':300,'test_rounds':200,
	'memory_batch_size': 512,'steps_warmup': 600,'steps_target_model_update':1000,'steps_per_iter':10000,'test_rounds':1000,
}

regime_params_final = {
	'learning_rate': 1e-6,
	#'memory_batch_size': 128,'steps_warmup': 100,'steps_target_model_update':100,'steps_per_iter':300,'test_rounds':500,
	'memory_batch_size': 1024,'steps_warmup': 600,'steps_target_model_update':1000,'steps_per_iter':10000,'test_rounds':1000,
}


regime_params_eternal = {
	'learning_rate': 1e-8,
	#'memory_batch_size': 128,'steps_warmup': 100,'steps_target_model_update':100,'steps_per_iter':300,'test_rounds':500,
	'memory_batch_size': 2048,'steps_warmup': 600,'steps_target_model_update':1000,'steps_per_iter':10000,'test_rounds':1000,
}


regime_params_cust = {
	'epsilon-train':0.1,
	'epsilon-chaos':0.1,
	'learning_rate': 1e-4,
	'memory_batch_size': 1024,'steps_warmup': 100,'steps_target_model_update':200,'steps_per_iter':1000,'test_rounds':700,
}

def merge_two_dicts(x, y):
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z


# = regime_params_wakeup
#r = regime_params_early
r = regime_params_late
#r = regime_params_final
#r = regime_params_eternal
#r = regime_params_cust


regime_params = merge_two_dicts( regime_params_base,r )
