import pandas as pd
import os
from utilities.utility_DualProcess import DualProcessModel
from utilities.utility_ComputationalModeling import ComputationalModels
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


# after the simulation has been completed, we can just load the simulated data from the folder
folder_path = './data/DataFitting/FittingResults'
fitting_results = {}

for file in os.listdir(folder_path):
    if file.endswith('.csv'):  # Check if the file is a CSV
        file_path = os.path.join(folder_path, file)  # Get the full path of the file
        df_name = os.path.splitext(file)[0]  # Extract the file name without the extension
        fitting_results[df_name] = pd.read_csv(file_path)

# unnest the dictionary into dfs
for key in fitting_results:
    globals()[key] = fitting_results[key]


# post hoc simulation
model = DualProcessModel()
model_decay = ComputationalModels(model_type='decay')
model_delta = ComputationalModels(model_type='delta')
model_actr = ComputationalModels(model_type='ACTR')

# simulate the data
LV_results = [Dual_LV_results]
MV_results = [Dual_MV_results]
HV_results = [Dual_HV_results]
models = ['Dual']
decay_results = [decay_LV_results, decay_MV_results, decay_HV_results]
delta_results = [delta_LV_results, delta_MV_results, delta_HV_results]
actr_results = [actr_LV_results, actr_MV_results, actr_HV_results]

reward_means = [0.65, 0.35, 0.75, 0.25]
hv = [0.48, 0.48, 0.43, 0.43]
mv = [0.24, 0.24, 0.22, 0.22]
lv = [0.12, 0.12, 0.11, 0.11]
sd = [lv, mv, hv]
df = [LV_df, MV_df, HV_df]


for i in range(len(LV_results)):
    file_name = f'./data/Post_hoc/{models[i]}_posthoc_LV.csv'
    if not os.path.exists(file_name):
        print(f"Simulating {models[i]}_posthoc_LV.csv")
        simulated_data = model.post_hoc_simulation(LV_results[i], LV_df, models[i],
                                                   reward_means, lv, num_iterations=10000)
        simulated_data.to_csv(file_name, index=False)
    else:
        print(f"{models[i]}_posthoc_LV.csv already exists")

for i in range(len(MV_results)):
    file_name = f'./data/Post_hoc/{models[i]}_posthoc_MV.csv'
    if not os.path.exists(file_name):
        print(f"Simulating {models[i]}_posthoc_MV.csv")
        simulated_data = model.post_hoc_simulation(MV_results[i], MV_df, models[i],
                                                   reward_means, mv, num_iterations=10000)
        simulated_data.to_csv(file_name, index=False)
    else:
        print(f"{models[i]}_posthoc_MV.csv already exists")

for i in range(len(HV_results)):
    file_name = f'./data/Post_hoc/{models[i]}_posthoc_HV.csv'
    if not os.path.exists(file_name):
        print(f"Simulating {models[i]}_posthoc_HV.csv")
        simulated_data = model.post_hoc_simulation(HV_results[i], HV_df, models[i],
                                                   reward_means, hv, num_iterations=10000)
        simulated_data.to_csv(file_name, index=False)
    else:
        print(f"{models[i]}_posthoc_HV.csv already exists")


# decay model
for i in range(len(decay_results)):
    simulated_data = model_decay.post_hoc_simulation(decay_results[i], df[i], reward_means,
                                                        sd[i], num_iterations=10000)
    simulated_data.to_csv(f'./data/Post_hoc/decay_posthoc_{df[i]["Condition"].unique()[0]}.csv', index=False)

# delta rule model
for i in range(len(delta_results)):
    simulated_data = model_delta.post_hoc_simulation(delta_results[i], df[i], reward_means,
                                                        sd[i], num_iterations=10000)
    simulated_data.to_csv(f'./data/Post_hoc/delta_posthoc_{df[i]["Condition"].unique()[0]}.csv', index=False)

# actr model
for i in range(len(actr_results)):
    simulated_data = model_actr.post_hoc_simulation(actr_results[i], df[i], reward_means,
                                                        sd[i], num_iterations=10000)
    simulated_data.to_csv(f'./data/Post_hoc/actr_posthoc_{df[i]["Condition"].unique()[0]}.csv', index=False)