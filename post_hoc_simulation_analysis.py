import numpy as np
import pandas as pd
import os
import ast
from utilities.utility_DataAnalysis import MSE_calculation

# import original data
data = pd.read_csv("./data/ABCDContRewardsAllData.csv")

LV = data[data['Condition'] == 'LV']
MV = data[data['Condition'] == 'MV']
HV = data[data['Condition'] == 'HV']

LV = LV.reset_index(drop=True)
MV = MV.reset_index(drop=True)
HV = HV.reset_index(drop=True)

# import post-hoc simulation results
folder_path = './data/Post_hoc'
simulations = {}

for file in os.listdir(folder_path):
    if file.endswith('.csv'):  # Check if the file is a CSV
        file_path = os.path.join(folder_path, file)  # Get the full path of the file
        df_name = os.path.splitext(file)[0]  # Extract the file name without the extension
        simulations[df_name] = pd.read_csv(file_path)

# unnest the dictionary into dfs
for key in simulations:
    if 'HV' in key:
        # combine the original trial sequence with the simulated data
        simulations[key] = pd.concat([HV[['bestOption', 'TrialType']], simulations[key]], axis=1)
        simulations[key]['Condition'] = 'HV'
    elif 'MV' in key:
        simulations[key] = pd.concat([MV[['bestOption', 'TrialType']], simulations[key]], axis=1)
        simulations[key]['Condition'] = 'MV'
    else:
        simulations[key] = pd.concat([LV[['bestOption', 'TrialType']], simulations[key]], axis=1)
        simulations[key]['Condition'] = 'LV'
    # now separate the key to find the model type
    model_type = key.split('_')[0]
    # add the model type to the dataframe
    simulations[key]['model'] = model_type

    # now, combine all the elements in the dictionary into a single dataframe
    if 'all_posthoc' not in locals():
        all_posthoc = simulations[key]
    else:
        all_posthoc = pd.concat([all_posthoc, simulations[key]], axis=0)

    globals()[key] = simulations[key]

# calculate the MSE for each model
for key, value in simulations.items():
    mse = MSE_calculation(value['bestOption'], value['choice'])
    print('Condition: {}; Model: {}; MSE: {}'.format(
        value['Condition'].unique()[0], key.split('_')[0], mse))

# participant level analysis
MSE_by_participant = all_posthoc.groupby(['Condition', 'model', 'Subnum'])[[
    'choice', 'bestOption']].apply(lambda x: MSE_calculation(x['bestOption'], x['choice'])).reset_index()
MSE_by_participant.columns = ['Condition', 'Model', 'Subnum', 'MSE']
# change the model column to categorical
MSE_by_participant['Model'] = pd.Categorical(MSE_by_participant['Model'], categories=['delta', 'decay', 'Dir',
                                                                                      'Gau', 'Dual'])
MSE_by_participant['Condition'] = pd.Categorical(MSE_by_participant['Condition'])

# mixed effects model
import statsmodels.api as sm
import statsmodels.formula.api as smf


data_mixed = MSE_by_participant[MSE_by_participant['Condition'] == 'HV']
# Create a mixed-effects model formula
formula = 'MSE ~ Model'

# Fit the mixed-effects model
mixed_model = smf.mixedlm(formula, data=data_mixed, groups='Subnum').fit()

# Print the summary of the model
print(mixed_model.summary())

x = HV.groupby('Subnum')['TrialType'].apply(list)