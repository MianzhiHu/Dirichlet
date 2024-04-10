import pandas as pd
from utilities.utility_DualProcess import DualProcessModel
from utilities.utility_ComputationalModeling import dict_generator, ComputationalModels, bayes_factor


if __name__ == '__main__':
    # load data
    data = pd.read_csv("./data/ABCDContRewardsAllData.csv")

    LV = data[data['Condition'] == 'LV']
    MV = data[data['Condition'] == 'MV']
    HV = data[data['Condition'] == 'HV']

    dataframes = [LV, MV, HV]
    for i in range(len(dataframes)):
        dataframes[i] = dataframes[i].reset_index(drop=True)
        dataframes[i].iloc[:, 1] = (dataframes[i].index // 250) + 1
        dataframes[i].rename(columns={'X': 'Subnum', 'setSeen': 'SetSeen.', 'choice': 'KeyResponse', 'points': 'Reward'}, inplace=True)
        dataframes[i]['KeyResponse'] = dataframes[i]['KeyResponse'] - 1
        dataframes[i]['SetSeen.'] = dataframes[i]['SetSeen.'] - 1
        dataframes[i] = dict_generator(dataframes[i])

    LV, MV, HV = dataframes

    model = DualProcessModel()
    decay = ComputationalModels("decay")
    delta = ComputationalModels("delta")

    # LV_result = model.fit(LV, 'Dual', num_iterations=100)
    # MV_result = model.fit(MV, 'Dual', num_iterations=100)
    # HV_result = model.fit(HV, 'Dual', num_iterations=100)
    #
    # # Save the results
    # LV_result.to_csv('./data/dual_LV_results.csv', index=False)
    # MV_result.to_csv('./data/dual_MV_results.csv', index=False)
    # HV_result.to_csv('./data/dual_HV_results.csv', index=False)
    #
    for model_type in ['Dir', 'Gau', 'Param']:
        result = model.fit(MV, model_type, num_iterations=100)
        result.to_csv(f'./data/DataFitting/{model_type}_MV_results.csv', index=False)

    # HV_decay = decay.fit(HV, num_iterations=100)
    # HV_delta = delta.fit(HV, num_iterations=100)
    # MV_decay = decay.fit(MV, num_iterations=100)
    # MV_delta = delta.fit(MV, num_iterations=100)
    # LV_decay = decay.fit(LV, num_iterations=100)
    # LV_delta = delta.fit(LV, num_iterations=100)

    # # save
    # HV_decay.to_csv('./data/DataFitting/decay_HV_results.csv', index=False)
    # HV_delta.to_csv('./data/DataFitting/delta_HV_results.csv', index=False)
    # MV_decay.to_csv('./data/DataFitting/decay_MV_results.csv', index=False)
    # MV_delta.to_csv('./data/DataFitting/delta_MV_results.csv', index=False)
    # LV_decay.to_csv('./data/DataFitting/decay_LV_results.csv', index=False)
    # LV_delta.to_csv('./data/DataFitting/delta_LV_results.csv', index=False)


# # calculate the mean AIC and BIC values
# print(f"LV_decay AIC: {LV_decay['AIC'].mean()}")
# print(f"LV_decay BIC: {LV_decay['BIC'].mean()}")
# print(f"LV_delta AIC: {LV_delta['AIC'].mean()}")
# print(f"LV_delta BIC: {LV_delta['BIC'].mean()}")
# print(f"MV_decay AIC: {MV_decay['AIC'].mean()}")
# print(f"MV_decay BIC: {MV_decay['BIC'].mean()}")
# print(f"MV_delta AIC: {MV_delta['AIC'].mean()}")
# print(f"MV_delta BIC: {MV_delta['BIC'].mean()}")
# print(f"HV_decay AIC: {HV_decay['AIC'].mean()}")
# print(f"HV_decay BIC: {HV_decay['BIC'].mean()}")
# print(f"HV_delta AIC: {HV_delta['AIC'].mean()}")
# print(f"HV_delta BIC: {HV_delta['BIC'].mean()}")

# # calculate the Bayes factor
# LV_bf = bayes_factor(LV_decay, LV_delta)
# MV_bf = bayes_factor(MV_decay, MV_delta)
# HV_bf = bayes_factor(HV_decay, HV_delta)
