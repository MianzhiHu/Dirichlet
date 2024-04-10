import pandas as pd
import cProfile
import pstats
from utilities.utility_DualProcess import DualProcessModel
from utilities.utility_ComputationalModeling import dict_generator, ComputationalModels

# When running the model fitting function, I found out that it was taking too much time to run.
# So this file is to optimize the model fitting function to make it run faster and it serves no other purpose.
if __name__ == '__main__':

    profiler = cProfile.Profile()
    profiler.enable()

    # setting up the model
    model = DualProcessModel()

    reward_means = [0.65, 0.35, 0.75, 0.25]
    hv = [0.43, 0.43, 0.43, 0.43]
    mv = [0.265, 0.265, 0.265, 0.265]
    lv = [0.1, 0.1, 0.1, 0.1]
    uncertainty = [0.43, 0.43, 0.12, 0.12]

    RLmodel = ComputationalModels('decay')
    # model.simulate(reward_means, hv, model='Dual', AB_freq=100, CD_freq=50, num_iterations=30)

    # optimize the model fitting function
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
        # take only the Subnum 1-15
        dataframes[i] = dataframes[i][dataframes[i]['Subnum'] < 16]
        dataframes[i] = dict_generator(dataframes[i])

    LV, MV, HV = dataframes

    # LV_result = model.fit(LV, 'Dual', num_iterations=1)
    LV_result = RLmodel.fit(LV, num_iterations=20)

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats()
