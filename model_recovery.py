import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from utils.DualProcess import DualProcessModel
from utils.ComputationalModeling import ComputationalModels, model_recovery


if __name__ == '__main__':
    # define the models
    model = DualProcessModel()
    model_decay = ComputationalModels(model_type='decay')
    model_delta = ComputationalModels(model_type='delta')
    model_actr = ComputationalModels(model_type='ACTR')
    model_utility = ComputationalModels(model_type='mean_var_utility')
    model_asym = ComputationalModels(model_type='delta_asymmetric')

    # define the reward means and variances
    reward_means = [0.65, 0.35, 0.75, 0.25]
    hv = [0.48, 0.48, 0.43, 0.43]
    mv = [0.24, 0.24, 0.22, 0.22]
    lv = [0.12, 0.12, 0.11, 0.11]
    n = 100
    n_fitting_iterations = 30

    hv_sim, hv_fit, hv_recovery = model_recovery(
        ['Dual-Process', 'Delta', 'Risk-Sensitive Delta', 'Mean-Variance Utility', 'Decay', 'ACT-R'],
        [model, model_delta, model_asym, model_utility, model_decay, model_actr],
        reward_means, hv, n_iterations=n, n_fitting_iterations=n_fitting_iterations)

    hv_sim.to_csv('./data/Simulation/Model Recovery/hv_sim.csv', index=False)
    hv_fit.to_csv('./data/Simulation/Model Recovery/hv_fit.csv', index=False)
    hv_recovery.to_csv('./data/Simulation/Model Recovery/hv_recovery.csv', index=False)

    mv_sim, mv_fit, mv_recovery = model_recovery(
        ['Dual-Process', 'Delta', 'Risk-Sensitive Delta', 'Mean-Variance Utility', 'Decay', 'ACT-R'],
        [model, model_delta, model_asym, model_utility, model_decay, model_actr],
        reward_means, mv, n_iterations=n, n_fitting_iterations=n_fitting_iterations)

    mv_sim.to_csv('./data/Simulation/Model Recovery/mv_sim.csv', index=False)
    mv_fit.to_csv('./data/Simulation/Model Recovery/mv_fit.csv', index=False)
    mv_recovery.to_csv('./data/Simulation/Model Recovery/mv_recovery.csv', index=False)

    lv_sim, lv_fit, lv_recovery = model_recovery(
        ['Dual-Process', 'Delta', 'Risk-Sensitive Delta', 'Mean-Variance Utility', 'Decay', 'ACT-R'],
        [model, model_delta, model_asym, model_utility, model_decay, model_actr],
        reward_means, lv, n_iterations=n, n_fitting_iterations=n_fitting_iterations)

    lv_sim.to_csv('./data/Simulation/Model Recovery/lv_sim.csv', index=False)
    lv_fit.to_csv('./data/Simulation/Model Recovery/lv_fit.csv', index=False)
    lv_recovery.to_csv('./data/Simulation/Model Recovery/lv_recovery.csv', index=False)

    # # construct the confusion and inversion matrix
    # model_recovery_df = hv_model_recovery
    # m_confusion = confusion_matrix(model_recovery_df['simulate_model'], model_recovery_df['fit_model'], normalize='true')
    # m_inversion = confusion_matrix(model_recovery_df['simulate_model'], model_recovery_df['fit_model'], normalize='pred')
    #
    # # plot the confusion matrix
    # sns.heatmap(m_inversion, annot=True)
    # plt.xlabel('Predicted')
    # plt.ylabel('Actual')
    # plt.savefig('./figures/model_recovery.png')
    # plt.show()


