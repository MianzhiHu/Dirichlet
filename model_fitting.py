import numpy as np
import pandas as pd
import os
from scipy.stats import dirichlet, multivariate_normal, entropy, norm, differential_entropy
from utilities.utility_DualProcess import DualProcessModel
from utilities.utility_ComputationalModeling import dict_generator, ComputationalModels, bayes_factor

if __name__ == '__main__':

    data = pd.read_csv("./data/ABCDContRewardsAllData.csv")

    LV = data[data['Condition'] == 'LV']
    MV = data[data['Condition'] == 'MV']
    HV = data[data['Condition'] == 'HV']

    dataframes = [LV, MV, HV]
    for i in range(len(dataframes)):
        dataframes[i] = dataframes[i].reset_index(drop=True)
        dataframes[i].iloc[:, 1] = (dataframes[i].index // 250) + 1
        dataframes[i].rename(
            columns={'X': 'Subnum', 'setSeen': 'SetSeen.', 'choice': 'KeyResponse', 'points': 'Reward'}, inplace=True)
        dataframes[i]['KeyResponse'] = dataframes[i]['KeyResponse'] - 1
        dataframes[i]['SetSeen.'] = dataframes[i]['SetSeen.'] - 1
        dataframes[i]['trial_index'] = dataframes[i].groupby('Subnum').cumcount() + 1

    LV_df, MV_df, HV_df = dataframes

    dicts = [LV_df, MV_df, HV_df]
    for i in range(len(dicts)):
        dicts[i] = dict_generator(dicts[i])

    LV, MV, HV = dicts

    # fit uncertainty data
    uncertainty_data = pd.read_csv('./data/UncertaintyData.csv')
    uncertainty_data['KeyResponse'] = uncertainty_data['KeyResponse'] - 1
    uncertainty = dict_generator(uncertainty_data)

    model = DualProcessModel()
    decay = ComputationalModels("decay")
    delta = ComputationalModels("delta")
    actr = ComputationalModels("ACTR")

    # this is for testing
    # select the first 500 rows for testing
    # testing_data = HV_df.iloc[:250, :]
    # testing_data = HV_df.iloc[501:510, :]
    testing_data = HV_df
    testing_data = dict_generator(testing_data)
    result = model.fit(testing_data, 'Entropy_Dis', num_iterations=150, weight_Gau='softmax', weight_Dir='softmax',
                       arbi_option='Entropy', Dir_fun='Normal', Gau_fun='Bayesian_Recency')
    # result = model.fit(testing_data, 'Recency', num_iterations=300, weight_Gau='softmax', arbi_option='Entropy',
    #                    weight_Dir='weight', Dir_fun='Normal', Gau_fun='Naive_Recency')
    # result_delta = delta.fit(testing_data, num_iterations=1)
    print(result['AIC'].mean())
    print(result['BIC'].mean())

    # print(result_delta['AIC'].mean())
    # print(result_delta['BIC'].mean())

    # result.to_csv('./data/DataFitting/FittingResults/BayesianRecencySW_HV_results.csv', index=False)

    # ==================================================================================================================
    # Model fitting starts here
    # ==================================================================================================================
    # fitting_models = ['Dir', 'Gau', 'Dual', 'Param', 'Multi_Param', 'Recency', 'Threshold', 'Recency_Threshold']
    # fitting_models = ['Recency']
    # Gau_fun = ['Bayesian_Recency', 'Naive_Recency', 'Bayesian', 'Naive']
    # Dir_fun = ['Linear_Recency', 'Normal']
    # Gau_weight = ['softmax']
    # Dir_weight = ['weight', 'softmax']
    #
    # for model_type in fitting_models:
    #     for gau_fun in Gau_fun:
    #         for dir_fun in Dir_fun:
    #             for gau_weight in Gau_weight:
    #                 for dir_weight in Dir_weight:
    #                     file_path = (f'./data/DataFitting/FittingResults/AllCombinations/{model_type}{gau_fun}{dir_fun}'
    #                                  f'{gau_weight}{dir_weight}_HV_results.csv')
    #                     if os.path.exists(file_path):
    #                         print(f'{model_type}_{gau_fun}_{dir_fun}_{gau_weight}_{dir_weight}_HV_results.csv already exists')
    #                     else:
    #                         result = model.fit(HV, model_type, num_iterations=100, weight_Gau=gau_weight, weight_Dir=dir_weight,
    #                                            arbi_option='Entropy', Dir_fun=dir_fun, Gau_fun=gau_fun)
    #                         result.to_csv(file_path, index=False)

    # for model_type in fitting_models:
    #     file_path = f'./data/DataFitting/FittingResults/DirWeightGauWeight/{model_type}_MV_results.csv'
    #     if os.path.exists(file_path):
    #         print(f'{model_type}_MV_results.csv already exists')
    #     else:
    #         result = model.fit(MV, model_type, num_iterations=300)
    #         result.to_csv(file_path, index=False)
    #
    # for model_type in fitting_models:
    #     file_path = f'./data/DataFitting/FittingResults/DirWeightGauWeight/{model_type}_LV_results.csv'
    #     if os.path.exists(file_path):
    #         print(f'{model_type}_LV_results.csv already exists')
    #     else:
    #         result = model.fit(LV, model_type, num_iterations=300)
    #         result.to_csv(file_path, index=False)

    # # ============== Naive =================
    # for model_type in ['Entropy', 'Confidence']:
    #     file_path = f'./data/DataFitting/FittingResults/{model_type}_HV_Naive_results.csv'
    #     if os.path.exists(file_path):
    #         print(f'{model_type}_HV_Naive_results.csv already exists')
    #     else:
    #         result = model.fit(HV, model_type, num_iterations=150, Gau_fun='Naive')
    #         result.to_csv(file_path, index=False)
    #
    # for model_type in ['Entropy_Recency', 'Confidence_Recency']:
    #     file_path = f'./data/DataFitting/FittingResults/{model_type}_HV_NaiveR_results.csv'
    #     if os.path.exists(file_path):
    #         print(f'{model_type}_HV_NaiveR_results.csv already exists')
    #     else:
    #         result = model.fit(HV, model_type, num_iterations=150, Gau_fun='Naive_Recency', Dir_fun='Recency')
    #         result.to_csv(file_path, index=False)
    # ============================================

    # ============== Uncertainty =================
    # for model_type in fitting_models:
    #     file_path = f'./data/DataFitting/FittingResults/{model_type}_uncertaintyOld_results.csv'
    #     if os.path.exists(file_path):
    #         print(f'{model_type}_uncertaintyOld_results.csv already exists')
    #     else:
    #         result = model.fit(uncertainty, model_type, num_iterations=200)
    #         result.to_csv(file_path, index=False)
    #
    # uncertainty_decay = decay.fit(uncertainty, num_iterations=100)
    # uncertainty_delta = delta.fit(uncertainty, num_iterations=100)
    # ============================================

    # # fit the traditional delta, decay, and actr models
    # HV_decay = decay.fit(HV, num_iterations=100)
    # HV_delta = delta.fit(HV, num_iterations=100)
    # HV_actr = actr.fit(HV, num_iterations=150)
    # HV_actr.to_csv('./data/DataFitting/FittingResults/actr_HV_results.csv', index=False)
    #
    # # MV_decay = decay.fit(MV, num_iterations=100)
    # # MV_delta = delta.fit(MV, num_iterations=100)
    # MV_actr = actr.fit(MV, num_iterations=150)
    # MV_actr.to_csv('./data/DataFitting/FittingResults/actr_MV_results.csv', index=False)
    # # LV_decay = decay.fit(LV, num_iterations=100)
    # # LV_delta = delta.fit(LV, num_iterations=100)
    # LV_actr = actr.fit(LV, num_iterations=150)
    # LV_actr.to_csv('./data/DataFitting/FittingResults/actr_LV_results.csv', index=False)

    #
    # # save
    # HV_decay.to_csv('./data/DataFitting/FittingResults/decay_HV_results.csv', index=False)
    # HV_delta.to_csv('./data/DataFitting/FittingResults/delta_HV_results.csv', index=False)
    # HV_actr.to_csv('./data/DataFitting/FittingResults/actr_HV_results.csv', index=False)
    # MV_decay.to_csv('./data/DataFitting/FittingResults/decay_MV_results.csv', index=False)
    # MV_delta.to_csv('./data/DataFitting/FittingResults/delta_MV_results.csv', index=False)
    # MV_actr.to_csv('./data/DataFitting/FittingResults/actr_MV_results.csv', index=False)
    # LV_decay.to_csv('./data/DataFitting/FittingResults/decay_LV_results.csv', index=False)
    # LV_delta.to_csv('./data/DataFitting/FittingResults/delta_LV_results.csv', index=False)
    # LV_actr.to_csv('./data/DataFitting/FittingResults/actr_LV_results.csv', index=False)
