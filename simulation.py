from utilities.utility_DualProcess import DualProcessModel

model = DualProcessModel()
reward_means = [0.65, 0.35, 0.75, 0.25]
hv = [0.43, 0.43, 0.43, 0.43]
mv = [0.265, 0.265, 0.265, 0.265]
lv = [0.1, 0.1, 0.1, 0.1]
uncertainty = [0.12, 0.12, 0.43, 0.43]

dir_hv = model.simulate(reward_means, hv, model='Dir', AB_freq=100, CD_freq=50)
dir_mv = model.simulate(reward_means, mv, model='Dir', AB_freq=100, CD_freq=50)
dir_lv = model.simulate(reward_means, lv, model='Dir', AB_freq=100, CD_freq=50)

gau_hv = model.simulate(reward_means, hv, model='Gau', AB_freq=100, CD_freq=50)
gau_mv = model.simulate(reward_means, mv, model='Gau', AB_freq=100, CD_freq=50)
gau_lv = model.simulate(reward_means, lv, model='Gau', AB_freq=100, CD_freq=50)

dual_hv = model.simulate(reward_means, hv, model='Dual', AB_freq=100, CD_freq=50)
dual_mv = model.simulate(reward_means, mv, model='Dual', AB_freq=100, CD_freq=50)
dual_lv = model.simulate(reward_means, lv, model='Dual', AB_freq=100, CD_freq=50)

mixed_hv = model.simulate(reward_means, hv, model='Param', AB_freq=100, CD_freq=50)
mixed_mv = model.simulate(reward_means, mv, model='Param', AB_freq=100, CD_freq=50)
mixed_lv = model.simulate(reward_means, lv, model='Param', AB_freq=100, CD_freq=50)

# uncertainty_hv = model.simulate(reward_means, uncertainty, model='Dual', AB_freq=100, CD_freq=50)
# uncertainty_mv = model.simulate(reward_means, uncertainty, model='Dual', AB_freq=100, CD_freq=50)
# uncertainty_lv = model.simulate(reward_means, uncertainty, model='Dual', AB_freq=100, CD_freq=50)
#
# # save the simulation results
# uncertainty_hv.to_csv('./data/Simulation/dual_uncertainty_hv.csv', index=False)
# uncertainty_mv.to_csv('./data/Simulation/dual_uncertainty_mv.csv', index=False)
# uncertainty_lv.to_csv('./data/Simulation/dual_uncertainty_lv.csv', index=False)

