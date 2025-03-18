import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import confusion_matrix
from utils.DualProcess import DualProcessModel
from utils.ComputationalModeling import ComputationalModels, model_recovery, parameter_extractor, dict_generator


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

    # hv_sim, hv_fit, hv_recovery = model_recovery(
    #     ['Dual-Process', 'Delta', 'Risk-Sensitive Delta', 'Mean-Variance Utility', 'Decay', 'ACT-R'],
    #     [model, model_delta, model_asym, model_utility, model_decay, model_actr],
    #     reward_means, hv, n_iterations=n, n_fitting_iterations=n_fitting_iterations)
    #
    # hv_sim.to_csv('./data/Simulation/Model Recovery/hv_sim.csv', index=False)
    # hv_fit.to_csv('./data/Simulation/Model Recovery/hv_fit.csv', index=False)
    # hv_recovery.to_csv('./data/Simulation/Model Recovery/hv_recovery.csv', index=False)
    #
    # mv_sim, mv_fit, mv_recovery = model_recovery(
    #     ['Dual-Process', 'Delta', 'Risk-Sensitive Delta', 'Mean-Variance Utility', 'Decay', 'ACT-R'],
    #     [model, model_delta, model_asym, model_utility, model_decay, model_actr],
    #     reward_means, mv, n_iterations=n, n_fitting_iterations=n_fitting_iterations)
    #
    # mv_sim.to_csv('./data/Simulation/Model Recovery/mv_sim.csv', index=False)
    # mv_fit.to_csv('./data/Simulation/Model Recovery/mv_fit.csv', index=False)
    # mv_recovery.to_csv('./data/Simulation/Model Recovery/mv_recovery.csv', index=False)
    #
    # lv_sim, lv_fit, lv_recovery = model_recovery(
    #     ['Dual-Process', 'Delta', 'Risk-Sensitive Delta', 'Mean-Variance Utility', 'Decay', 'ACT-R'],
    #     [model, model_delta, model_asym, model_utility, model_decay, model_actr],
    #     reward_means, lv, n_iterations=n, n_fitting_iterations=n_fitting_iterations)
    #
    # lv_sim.to_csv('./data/Simulation/Model Recovery/lv_sim.csv', index=False)
    # lv_fit.to_csv('./data/Simulation/Model Recovery/lv_fit.csv', index=False)
    # lv_recovery.to_csv('./data/Simulation/Model Recovery/lv_recovery.csv', index=False)


    # ==================================================================================================================
    # Model Recovery Analysis
    # ==================================================================================================================
    hv_model_recovery = pd.read_csv('./data/Simulation/Model Recovery/hv_recovery.csv')
    mv_model_recovery = pd.read_csv('./data/Simulation/Model Recovery/mv_recovery.csv')
    lv_model_recovery = pd.read_csv('./data/Simulation/Model Recovery/lv_recovery.csv')

    # construct the confusion and inversion matrix
    model_recovery_df = lv_model_recovery
    print(model_recovery_df['simulated_model'].unique())
    m_confusion = confusion_matrix(model_recovery_df['simulated_model'], model_recovery_df['fit_model'],
                                   normalize='true', labels=model_recovery_df['simulated_model'].unique())
    m_inversion = confusion_matrix(model_recovery_df['simulated_model'], model_recovery_df['fit_model'], normalize='pred',
                                   labels=model_recovery_df['simulated_model'].unique())

    dual = model_recovery_df[model_recovery_df['simulated_model'] == 'Dual-Process']
    print(f'Model Recovery for Dual-Process: {dual["fit_model"].value_counts()}')

    # plot the confusion matrix
    matrix_of_interest = m_inversion
    sns.heatmap(matrix_of_interest, annot=True, cmap='Blues', xticklabels=model_recovery_df['simulated_model'].unique(),
                yticklabels=model_recovery_df['simulated_model'].unique(), cbar=False)
    plt.xlabel('Fit Model')
    plt.ylabel('Simulated Model')
    plt.tight_layout()
    plt.savefig('./figures/model_recovery.png', dpi=600)
    plt.show()

    # model recovery 2
    hv_fit = pd.read_csv('./data/Simulation/Model Recovery/hv_fit.csv')
    mv_fit = pd.read_csv('./data/Simulation/Model Recovery/mv_fit.csv')
    lv_fit = pd.read_csv('./data/Simulation/Model Recovery/lv_fit.csv')
    fit = lv_fit

    # add 6 * i to the participant id every 36 rows
    fit['block'] = fit.index // 36
    fit['participant_id'] = (fit.index % 6 + 1) + 6 * fit['block']
    best_fitting_model = fit.loc[fit.groupby('participant_id')['AIC'].idxmin()].reset_index(drop=True)

    m_confusion = confusion_matrix(best_fitting_model['simulated_model'], best_fitting_model['fit_model'],
                                   normalize='true', labels=best_fitting_model['simulated_model'].unique())
    m_inversion = confusion_matrix(best_fitting_model['simulated_model'], best_fitting_model['fit_model'], normalize='pred',
                                   labels=best_fitting_model['simulated_model'].unique())

    dual = best_fitting_model[best_fitting_model['simulated_model'] == 'Dual-Process']
    print(f'Model Recovery for Dual-Process: {dual["fit_model"].value_counts()}')

    # plot the confusion matrix
    sns.heatmap(m_confusion, annot=True, cmap='Blues', xticklabels=best_fitting_model['simulated_model'].unique(),
                yticklabels=best_fitting_model['simulated_model'].unique(), cbar=False)
    plt.xlabel('Fit Model')
    plt.ylabel('Simulated Model')
    plt.tight_layout()
    plt.savefig('./figures/model_recovery.png', dpi=600)
    plt.show()

    # ==================================================================================================================
    # Parameter Recovery Analysis
    # ==================================================================================================================
    hv_fit = pd.read_csv('./data/Simulation/Model Recovery/hv_fit.csv')
    hv_sim = pd.read_csv('./data/Simulation/Model Recovery/hv_sim.csv')
    mv_fit = pd.read_csv('./data/Simulation/Model Recovery/mv_fit.csv')
    mv_sim = pd.read_csv('./data/Simulation/Model Recovery/mv_sim.csv')
    lv_fit = pd.read_csv('./data/Simulation/Model Recovery/lv_fit.csv')
    lv_sim = pd.read_csv('./data/Simulation/Model Recovery/lv_sim.csv')

    # add condition labels
    hv_fit.loc[:, 'Condition'] = 'HV'
    mv_fit.loc[:, 'Condition'] = 'MV'
    lv_fit.loc[:, 'Condition'] = 'LV'
    hv_sim.loc[:, 'Condition'] = 'HV'
    mv_sim.loc[:, 'Condition'] = 'MV'
    lv_sim.loc[:, 'Condition'] = 'LV'

    model_sim_df = pd.concat([hv_sim, mv_sim, lv_sim], axis=0)
    model_fit_df = pd.concat([hv_fit, mv_fit, lv_fit], axis=0)

    # extract the parameters
    dual_fit_df = model_fit_df[(model_fit_df['fit_model'] == 'Dual-Process') &
                               (model_fit_df['simulated_model'] == 'Dual-Process')]
    dual_fit_params = parameter_extractor(dual_fit_df).reset_index(drop=True)
    dual_fit_params.loc[:, 'participant_id'] = dual_fit_params.index + 1
    dual_fit_params.rename(columns={'participant_id': 'Subnum', 't': 't_fit', 'alpha': 'alpha_fit',
                                    'subj_weight': 'subj_weight_fit'}, inplace=True)

    # remove redundant rows in the simulated data
    dual_sim_df = model_sim_df[model_sim_df['simulated_model'] == 'Dual-Process'].reset_index(drop=True)
    dual_sim_df.loc[:, 'Subnum'] = dual_sim_df.index // 250 + 1
    dual_sim_params = dual_sim_df.groupby('Subnum').agg(
        t=('t', 'mean'),
        alpha=('a', 'mean'),
        subj_weight=('param_weight', 'mean'),
    ).reset_index()

    # calculate the correlation
    print(f't corr: {pearsonr(dual_fit_params["t_fit"], dual_sim_params["t"])}')
    print(f'a corr: {pearsonr(dual_fit_params["alpha_fit"], dual_sim_params["alpha"])}')
    print(f'weight corr: {pearsonr(dual_fit_params["subj_weight_fit"], dual_sim_params["subj_weight"])}')

    # visualize the parameter recovery
    combined_params = pd.concat([dual_fit_params, dual_sim_params], axis=1)

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    sns.regplot(x='t', y='t_fit', data=combined_params, ax=ax[0], scatter=False, color=sns.color_palette()[0])
    sns.scatterplot(x='t', y='t_fit', hue='Condition', style='Condition', data=combined_params, ax=ax[0],
                    palette=sns.color_palette()[1:4], alpha=0.5, legend=True)
    ax[0].set_xlabel('Simulated c')
    ax[0].set_ylabel('Best-Fitting c')

    sns.regplot(x='alpha', y='alpha_fit', data=combined_params, ax=ax[1], scatter=False, color=sns.color_palette()[0])
    sns.scatterplot(x='alpha', y='alpha_fit', hue='Condition', style='Condition', data=combined_params, ax=ax[1],
                    palette=sns.color_palette()[1:4], alpha=0.5, legend=False)
    ax[1].set_xlabel('Simulated Alpha')
    ax[1].set_ylabel('Best-Fitting Alpha')

    sns.regplot(x='subj_weight', y='subj_weight_fit', data=combined_params, ax=ax[2], scatter=False,
                color=sns.color_palette()[0])
    sns.scatterplot(x='subj_weight', y='subj_weight_fit', hue='Condition', style='Condition', data=combined_params,
                    ax=ax[2], palette=sns.color_palette()[1:4], alpha=0.5, legend=False)
    ax[2].set_xlabel('Simulated Subjective Dirichlet Weight')
    ax[2].set_ylabel('Best-fitting Subjective Dirichlet Weight')

    plt.tight_layout()
    plt.savefig('./figures/parameter_recovery.png', dpi=600)
    plt.show()



