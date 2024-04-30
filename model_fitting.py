import pandas as pd
import os
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

    # # this is for testing
    # # select the first 500 rows for testing
    # testing_data = LV_df.iloc[:500, :]
    # testing_data = dict_generator(testing_data)
    # result = model.fit(LV, 'Multi_Param', num_iterations=2)

    # for model_type in ['Dir', 'Gau', 'Dual', 'Param']:
    #     result = model.fit(HV, model_type, num_iterations=100)
    #     result.to_csv(f'./data/DataFitting/FittingResults/{model_type}_HV_results.csv', index=False)
    #
    # for model_type in ['Dir', 'Gau', 'Dual', 'Param']:
    #     result = model.fit(MV, model_type, num_iterations=100)
    #     result.to_csv(f'./data/DataFitting/FittingResults/{model_type}_MV_results.csv', index=False)
    #
    # for model_type in ['Dir', 'Gau', 'Dual', 'Param']:
    #     result = model.fit(LV, model_type, num_iterations=100)
    #     result.to_csv(f'./data/DataFitting/FittingResults/{model_type}_LV_results.csv', index=False)
    #
    # for model_type in ['Param']:
    #     result = model.fit(uncertainty, model_type, num_iterations=100)
    #     result.to_csv(f'./data/DataFitting/FittingResults/{model_type}Scale_uncertaintyOld_results.csv', index=False)

    # refit the param model with multiple parameters
    HV_param = model.fit(HV, 'Multi_Param', num_iterations=100)
    HV_param.to_csv('./data/DataFitting/FittingResults/MultiParam_HV_results.csv', index=False)

    MV_param = model.fit(MV, 'Multi_Param', num_iterations=100)
    MV_param.to_csv('./data/DataFitting/FittingResults/MultiParam_MV_results.csv', index=False)

    LV_param = model.fit(LV, 'Multi_Param', num_iterations=100)
    LV_param.to_csv('./data/DataFitting/FittingResults/MultiParam_LV_results.csv', index=False)

    # # # save
    # HV_param.to_csv('./data/DataFitting/FittingResults/ParamScale_HV_results.csv', index=False)
    # MV_param.to_csv('./data/DataFitting/FittingResults/ParamScale_MV_results.csv', index=False)
    # LV_param.to_csv('./data/DataFitting/FittingResults/ParamScale_LV_results.csv', index=False)
    #
    # # fit the traditional delta and decay models
    # HV_decay = decay.fit(HV, num_iterations=100)
    # HV_delta = delta.fit(HV, num_iterations=100)
    # MV_decay = decay.fit(MV, num_iterations=100)
    # MV_delta = delta.fit(MV, num_iterations=100)
    # LV_decay = decay.fit(LV, num_iterations=100)
    # LV_delta = delta.fit(LV, num_iterations=100)
    # uncertainty_decay = decay.fit(uncertainty, num_iterations=100)
    # uncertainty_delta = delta.fit(uncertainty, num_iterations=100)
    #
    # # save
    # HV_decay.to_csv('./data/DataFitting/decay_HV_results.csv', index=False)
    # HV_delta.to_csv('./data/DataFitting/delta_HV_results.csv', index=False)
    # MV_decay.to_csv('./data/DataFitting/decay_MV_results.csv', index=False)
    # MV_delta.to_csv('./data/DataFitting/delta_MV_results.csv', index=False)
    # LV_decay.to_csv('./data/DataFitting/decay_LV_results.csv', index=False)
    # LV_delta.to_csv('./data/DataFitting/delta_LV_results.csv', index=False)
    # uncertainty_decay.to_csv('./data/DataFitting/FittingResults/decay_uncertaintyOld_results.csv', index=False)
    # uncertainty_delta.to_csv('./data/DataFitting/FittingResults/delta_uncertaintyOld_results.csv', index=False)
