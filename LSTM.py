import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error


# define the LSTM model
class LSTM(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, layer_num):
        super().__init__()
        self.lstmLayer = nn.LSTM(in_dim, hidden_dim, layer_num)
        self.relu = nn.ReLU()
        self.fcLayer = nn.Linear(hidden_dim, out_dim)
        self.weightInit = np.sqrt(1.0 / hidden_dim)

    def forward(self, x, mask):
        out, _ = self.lstmLayer(x)
        out = self.relu(out)
        out = self.fcLayer(out)
        # set the unavailable options to -inf so that the softmax function will ignore them
        out = torch.where(mask == 1, out, torch.tensor(float('-inf')))
        out = nn.Softmax(dim=-1)(out)
        return out


# define some needed functions
def encode_trial_type(df):
    # Create dummy columns for each option
    df['Option_A'] = 0
    df['Option_B'] = 0
    df['Option_C'] = 0
    df['Option_D'] = 0

    # Iterate through the DataFrame to set dummies
    for index, row in df.iterrows():
        if 'A' in row['TrialType']:
            df.at[index, 'Option_A'] = 1
        if 'B' in row['TrialType']:
            df.at[index, 'Option_B'] = 1
        if 'C' in row['TrialType']:
            df.at[index, 'Option_C'] = 1
        if 'D' in row['TrialType']:
            df.at[index, 'Option_D'] = 1

    return df


def find_trial_type(row):  # Function to convert the one-hot encoding back to the trial type
    options = ['A', 'B', 'C', 'D']
    trial_type = ''.join([options[i] for i in range(len(row)) if row[i] > 0])
    return trial_type


def find_best_choice(row, target_variable):  # Function to extract the value of the best choice from the numpy array

    best_choice_mapping = {
        'AB': 0,
        'CD': 2,
        'AC': 2,
        'AD': 0,
        'BC': 2,
        'BD': 1,
    }

    # extract the value from the position to indicate the best choice
    best_choice_position = best_choice_mapping[row['TrialType']]
    best_choice = row[target_variable][best_choice_position]
    return best_choice


# =============================================================================
# The data preparation process starts here
# =============================================================================
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

# prepare data for LSTM
data = HV_df

# standardize the reward
data['Reward'] = (data['Reward'] - data['Reward'].mean()) / data['Reward'].std()

# set the reward to 0 after 150 trials
data.loc[data['trial_index'] > 150, 'Reward'] = 0

# add a column to indicate whether the reward is seen by the participant
data['RewardSeen'] = 1
data.loc[data['trial_index'] > 150, 'RewardSeen'] = 0

# Function to encode pairs
encode_map = {
    'AB': ['A', 'B'],
    'CD': ['C', 'D'],
    'CA': ['C', 'A'],
    'CB': ['C', 'B'],
    'BD': ['B', 'D'],
    'AD': ['A', 'D']
}

# Apply the encoding function
data = encode_trial_type(data)

# use dummy variables for the all the categorical variables
data = pd.get_dummies(data, columns=['KeyResponse'])
data[['KeyResponse_0', 'KeyResponse_1', 'KeyResponse_2', 'KeyResponse_3']] = data[
    ['KeyResponse_0', 'KeyResponse_1', 'KeyResponse_2', 'KeyResponse_3']].astype(int)

var = ['Reward', 'RewardSeen', 'Option_A', 'Option_B', 'Option_C', 'Option_D',
       'KeyResponse_0', 'KeyResponse_1', 'KeyResponse_2', 'KeyResponse_3']

# Group data by subject or another grouping variable if needed
grouped = data.groupby('Subnum').apply(lambda x: x[var].values.tolist())

sequences = list(grouped.values)
num_participants = len(sequences)
max_len = max([len(x) for x in sequences])
padded_sequences = torch.zeros((len(sequences), max_len, len(var)))

for i, seq in enumerate(sequences):
    for j, step in enumerate(seq):
        padded_sequences[i, j] = torch.tensor(step)

features = padded_sequences[:, :, :]
targets = padded_sequences[:, :, -4:]
mask = padded_sequences[:, :, 2:6]

# =============================================================================
# The model fitting process starts here
# =============================================================================

# define model parameters
n_folds = 5
lag = 1  # the model prediction starts from the second trial
n_layers = 3
n_nodes = 10
n_epochs = 10
batch_size = 1

# define the model
model = LSTM(len(var), n_nodes, 4, n_layers)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

# split the data into n_folds
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

# iterate over the folds
for fold, (train_index, test_index) in enumerate(kf.split(features)):
    # split the data
    X_train, X_test = features[train_index], features[test_index]
    y_train, y_test = targets[train_index], targets[test_index]
    mask_train, mask_test = mask[train_index], mask[test_index]

    losses = []

    for epoch in np.arange(n_epochs):
        for i in np.arange(X_train.shape[0] / batch_size):
            batch_participant_ids = train_index[int(i * batch_size):int((i + 1) * batch_size)]

            X_batch = X_train[int(i * batch_size):int((i + 1) * batch_size)].float()
            y_batch = y_train[int(i * batch_size):int((i + 1) * batch_size)].float()
            mask_batch = mask_train[int(i * batch_size):int((i + 1) * batch_size)].float()

            optimizer.zero_grad()
            output = model(X_batch, mask_batch)
            loss = criterion(output[:, :-lag], y_batch[:, lag:])
            loss.backward()

            on_policy_loss = loss.item()
            losses.append(on_policy_loss)

            optimizer.step()

            # print the number of folds
            if i % 10 == 0:
                print(f'Fold: {fold + 1}/{n_folds}')
                print('Epoch[{}/{}], Batch[{}/{}], Loss: {:.5f}'.format(
                    epoch + 1, n_epochs, i + 1, X_train.shape[0] / batch_size, on_policy_loss))

    # evaluate
    model_eval = model.eval()
    y_pred = model_eval(X_test, mask_test).data.cpu().numpy()

    if fold == 0:
        test_set_full = y_test
        pred_set_full = y_pred
        MSEloss = losses
    else:
        test_set_full = np.concatenate((test_set_full, y_test), axis=0)
        pred_set_full = np.concatenate((pred_set_full, y_pred), axis=0)
        MSEloss = np.concatenate((MSEloss, losses), axis=0)

# =============================================================================
# The model fitting process ends here
# =============================================================================

# convert the results back to the original format
for results in [test_set_full, pred_set_full]:
    participant = []
    trial_index = []
    outcome = []
    for i in range(results.shape[0]):
        for j in range(results.shape[1]):
            participant.append(i)
            trial_index.append(j)
            outcome.append(results[i, j])

    # if the result df doesn't exist, create it, otherwise, join the results
    if 'LSTM_result' not in locals():
        LSTM_result = pd.DataFrame({'Subnum': participant, 'trial_index': trial_index, 'real_y': outcome})
    else:
        LSTM_result = pd.merge(LSTM_result,
                               pd.DataFrame({'Subnum': participant, 'trial_index': trial_index, 'pred_y': outcome}),
                               on=['Subnum', 'trial_index'])


# now, get the trial type first
LSTM_result['TrialType'] = LSTM_result['pred_y'].apply(find_trial_type)


# get people's actual choices, as well as the predicted choices
LSTM_result['bestOption'] = LSTM_result.apply(find_best_choice, axis=1, target_variable='real_y')
LSTM_result['pred_bestOption'] = LSTM_result.apply(find_best_choice, axis=1, target_variable='pred_y')

# calculate the mean squared error
on_policy_MSE = np.mean(MSEloss)
MSE = np.mean((LSTM_result['bestOption'] - LSTM_result['pred_bestOption']) ** 2)
MAE = np.mean(np.abs(LSTM_result['bestOption'] - LSTM_result['pred_bestOption']))

# get the weights
weights = model.fcLayer.weight.data.cpu().numpy()

