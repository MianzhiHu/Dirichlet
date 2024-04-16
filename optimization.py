import pandas as pd
import numpy as np
import cProfile
import pstats
from utilities.utility_DualProcess import DualProcessModel
from utilities.utility_ComputationalModeling import dict_generator, ComputationalModels
from sklearn.linear_model import LogisticRegression

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
        dataframes[i].rename(
            columns={'X': 'Subnum', 'setSeen': 'SetSeen.', 'choice': 'KeyResponse', 'points': 'Reward'}, inplace=True)
        dataframes[i]['KeyResponse'] = dataframes[i]['KeyResponse'] - 1
        dataframes[i]['SetSeen.'] = dataframes[i]['SetSeen.'] - 1
        # take only the Subnum 1-15
        dataframes[i] = dataframes[i][dataframes[i]['Subnum'] < 16]
        dataframes[i] = dict_generator(dataframes[i])

    LV, MV, HV = dataframes

    # # LV_result = model.fit(LV, 'Dual', num_iterations=1)
    # LV_result = RLmodel.fit(LV, num_iterations=20)
    #
    # profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats('cumtime')
    # stats.print_stats()

    #############################################################################
    # The following is an attempt to calculate indifference points for the COVID-19 study

    # Example DataFrame structure
    data = pd.DataFrame({
        'Participant_ID': np.repeat(range(10), 100),  # 10 participants, 100 trials each
        'Lives_Saved': np.random.randint(10, 100, 1000),
        'Unemployment_Change': np.random.uniform(0.1, 2.0, 1000),
        'Response': np.random.choice([0, 1], 1000)
    })


    # Function to fit model and calculate indifference point
    def analyze_participant(participant_data):
        model = LogisticRegression()
        features = participant_data[['Lives_Saved', 'Unemployment_Change']]
        response = participant_data['Response']
        model.fit(features, response)

        # Coefficients
        intercept = model.intercept_[0]
        coef_lives = model.coef_[0][0]
        coef_unemp = model.coef_[0][1]

        # For A = 50 lives (example calculation)
        lives = 50
        indifference_unemployment_change = (intercept + coef_lives * lives) / coef_unemp
        return indifference_unemployment_change


    # Apply function to each participant
    indifference_points = data.groupby('Participant_ID').apply(analyze_participant)

    # Print results
    print(indifference_points)


