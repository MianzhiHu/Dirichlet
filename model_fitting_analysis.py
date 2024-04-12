import numpy as np
import pandas as pd
import os
from utilities.utility_DataAnalysis import mean_AIC_BIC, create_bayes_matrix

# after the simulation has been completed, we can just load the simulated data from the folder
folder_path = './data/DataFitting/FittingResults'
simulations = {}

for file in os.listdir(folder_path):
    if file.endswith('.csv'):  # Check if the file is a CSV
        file_path = os.path.join(folder_path, file)  # Get the full path of the file
        df_name = os.path.splitext(file)[0]  # Extract the file name without the extension
        simulations[df_name] = pd.read_csv(file_path)

# unnest the dictionary into dfs
for key in simulations:
    globals()[key] = simulations[key]


# calculate the mean AIC and BIC values
for key in simulations:
    print(f"Model: {key}")
    mean_AIC_BIC(simulations[key])


# create bayes factor matrices
# Filter simulations for HV, MV, and LV
simulations_HV = {k: v for k, v in simulations.items() if 'HV' in k}
simulations_MV = {k: v for k, v in simulations.items() if 'MV' in k}
simulations_LV = {k: v for k, v in simulations.items() if 'LV' in k}

# Create Bayes factor matrices for HV, MV, and LV
bayes_matrix_HV = create_bayes_matrix(simulations_HV, 'HV Bayes Factor Matrix')
bayes_matrix_MV = create_bayes_matrix(simulations_MV, 'MV Bayes Factor Matrix')
bayes_matrix_LV = create_bayes_matrix(simulations_LV, 'LV Bayes Factor Matrix')

# # explode ProcessChosen
# # Ensure the column is a list
# dual_MV_results['best_process_chosen'] = dual_MV_results['best_process_chosen'].apply(lambda x: x if isinstance(x, list) else eval(x))
#
# # Use explode
# process_chosen = dual_MV_results['best_process_chosen'].explode()
# HV_df['best_process_chosen'] = process_chosen.values
# process_chosen_MV = HV_df.groupby('TrialType')['best_process_chosen'].value_counts(normalize=True).unstack().fillna(0)


