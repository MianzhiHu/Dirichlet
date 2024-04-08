import pandas as pd
from utilities.utility_DualProcess import DualProcessModel
from utilities.utility_ComputationalModeling import dict_generator


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
# LV_result = model.fit(LV, 'Dual', num_iterations=100)
# MV_result = model.fit(MV, 'Dual', num_iterations=100)
# HV_result = model.fit(HV, 'Dual', num_iterations=100)

# # Save the results
# LV_result.to_csv('./data/LV_results_dual.csv', index=False)
# MV_result.to_csv('./data/MV_results_dual.csv', index=False)
# HV_result.to_csv('./data/HV_results_dual.csv', index=False)

for model_type in ['Dir', 'Gau', 'Param']:
    result = model.fit(HV, model_type, num_iterations=100)
    result.to_csv(f'./data/DataFitting/{model_type}_hv_results.csv', index=False)

