from utilities.utility_DualProcess import DualProcessModel

model = DualProcessModel()
reward_means = [0.65, 0.35, 0.75, 0.25]
reward_sd = [0.43, 0.43, 0.43, 0.43]

dir = model.simulate(reward_means, reward_sd, model="Dir", num_iterations=1, sim_trials=250, AB_freq=100, CD_freq=50)
gau = model.simulate(reward_means, reward_sd, model="Gau", num_iterations=1, sim_trials=250, AB_freq=100, CD_freq=50)
dual = model.simulate(reward_means, reward_sd, model="Dual", num_iterations=1, sim_trials=250, AB_freq=100, CD_freq=50)
param = model.simulate(reward_means, reward_sd, model="Param", num_iterations=1, sim_trials=250, AB_freq=100, CD_freq=50)