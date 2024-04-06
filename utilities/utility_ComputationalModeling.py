import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2


class ComputationalModels:
    def __init__(self, reward_means, reward_sd, model_type, condition="Gains", num_trials=250,
                 num_params=2):
        """
        Initialize the Model.

        Parameters:
        - reward_means: List of mean rewards for each option.
        - reward_sd: List of standard deviations for each option.
        - model_type: Type of the model.
        - condition: Condition of the model.
        - num_trials: Number of trials for the simulation.
        - num_params: Number of parameters for the model.
        """
        self.num_options = 4
        self.num_trials = num_trials
        self.num_params = num_params
        self.choices_count = np.zeros(self.num_options)
        self.condition = condition
        self.memory_weights = []
        self.choice_history = []
        self.reward_history = []
        self.AllProbs = []
        self.PE = []
        self.AV = 0

        self.t = None
        self.a = None
        self.b = None
        self.iteration = 0

        if self.condition == "Gains":
            self.EVs = np.full(self.num_options, 0.5)
        elif self.condition == "Losses":
            self.EVs = np.full(self.num_options, -0.5)
        elif self.condition == "Both":
            self.EVs = np.full(self.num_options, 0.0)

        # Reward structure
        self.reward_means = reward_means
        self.reward_sd = reward_sd

        # Model type
        self.model_type = model_type

    def reset(self):
        """
        Reset the model.
        """
        self.choices_count = np.zeros(self.num_options)
        self.memory_weights = []
        self.choice_history = []
        self.reward_history = []
        self.AllProbs = []

        if self.condition == "Gains":
            self.EVs = np.full(self.num_options, 0.5)
            self.AV = 0.5
        elif self.condition == "Losses":
            self.EVs = np.full(self.num_options, -0.5)
            self.AV = -0.5
        elif self.condition == "Both":
            self.EVs = np.full(self.num_options, 0)
            self.AV = 0.0

    def update(self, chosen, reward, trial):
        """
        Update EVs based on the choice, received reward, and trial number.

        Parameters:
        - chosen: Index of the chosen option.
        - reward: Reward received for the current trial.
        - trial: Current trial number.
        """
        if trial > 150:
            return self.EVs

        self.choices_count[chosen] += 1

        if self.model_type == 'decay':
            self.EVs[chosen] += reward
            self.EVs = self.EVs * (1 - self.a)

        elif self.model_type == 'decay_fre':
            multiplier = self.choices_count[chosen] ** (-self.b)
            self.EVs[chosen] += reward * multiplier
            self.EVs = self.EVs * (1 - self.a)

        elif self.model_type == 'decay_choice':
            self.EVs[chosen] += 1
            self.EVs = self.EVs * (1 - self.a)

        elif self.model_type == 'decay_win':
            # # if use PE instead of AV
            # prediction_error = reward - self.EVs[chosen]

            # if use AV instead of PE
            prediction_error = reward - self.AV
            weighted_PE = prediction_error * self.a
            self.AV += weighted_PE
            if prediction_error > 0:
                self.EVs[chosen] += 1

            self.EVs = self.EVs * (1 - self.a)

        elif self.model_type == 'delta_decay':
            prediction_error = reward - self.EVs[chosen]
            self.EVs[chosen] += self.a * prediction_error
            self.EVs = self.EVs * (1 - self.b)

        elif self.model_type == 'delta':
            prediction_error = reward - self.EVs[chosen]
            self.EVs[chosen] += self.a * prediction_error

        elif self.model_type == 'sampler_decay':

            if self.num_params == 2:
                self.b = self.a

            self.reward_history.append(reward)
            self.choice_history.append(chosen)
            self.memory_weights.append(1)

            # # use the following code if you want to use prediction errors as reward instead of actual rewards
            # prediction_error = self.a * (reward - self.EVs[chosen])
            # self.PE.append(prediction_error)

            # # use the following code if you want to use average reward as reward instead of actual rewards
            # prediction_error = reward - self.AV
            # weighted_PE = prediction_error * self.a
            # self.AV += weighted_PE
            # self.PE.append(weighted_PE)

            # Decay weights of past trials and EVs
            self.EVs = self.EVs * (1 - self.a)
            self.memory_weights = [w * (1 - self.b) for w in self.memory_weights]

            # Compute the probabilities from memory weights
            total_weight = sum(self.memory_weights)
            self.AllProbs = [w / total_weight for w in self.memory_weights]

            # Update EVs based on the samples from memory
            for j in range(len(self.reward_history)):
                self.EVs[self.choice_history[j]] += self.AllProbs[j] * self.reward_history[j]

            # # For PE and AV versions
            # for j in range(len(self.memory_weights)):
            #     self.EVs[self.choice_history[j]] += self.AllProbs[j] * self.PE[j]

        # print(f'C: {chosen}, R: {reward}, EV: {self.EVs}; it has been {self.choices_count[chosen]} times')
        return self.EVs

    def softmax(self, chosen, alt1):
        c = 3 ** self.t - 1
        num = np.exp(min(700, c * chosen))
        denom = num + np.exp(min(700, c * alt1))
        return num / denom

    def simulate(self, num_trials=250, AB_freq=None, CD_freq=None, num_iterations=1000,
                 beta_lower=-1, beta_upper=1):
        """
        Simulate the EV updates for a given number of trials and specified number of iterations.

        Parameters:
        - num_trials: Number of trials for the simulation.
        - AB_freq: Frequency of appearance for the AB pair in the first 150 trials.
        - CD_freq: Frequency of appearance for the CD pair in the first 150 trials.
        - num_iterations: Number of times to repeat the simulation.
        - beta_lower: Lower bound for the beta parameter.
        - beta_upper: Upper bound for the beta parameter.

        Returns:
        - A list of simulation results.
        """
        all_results = []

        for iteration in range(num_iterations):

            print(f"Iteration {iteration + 1} of {num_iterations}")

            self.t = np.random.uniform(0, 5)
            self.a = np.random.uniform()  # Randomly set decay parameter between 0 and 1
            self.b = np.random.uniform(beta_lower, beta_upper)
            self.choices_count = np.zeros(self.num_options)

            EV_history = np.zeros((num_trials, self.num_options))
            trial_details = []
            trial_indices = []

            training_trials = [(0, 1), (2, 3)]
            training_trial_sequence = [training_trials[0]] * AB_freq + [training_trials[1]] * CD_freq
            np.random.shuffle(training_trial_sequence)

            # Distributing the next 100 trials equally among the four pairs (AC, AD, BC, BD)
            transfer_trials = [(2, 0), (1, 3), (0, 3), (2, 1)]
            transfer_trial_sequence = transfer_trials * 25
            np.random.shuffle(transfer_trial_sequence)

            for trial in range(num_trials):
                trial_indices.append(trial + 1)

                if trial < 150:
                    pair = training_trial_sequence[trial]
                    optimal, suboptimal = (pair[0], pair[1])
                    prob_optimal = self.softmax(self.EVs[optimal], self.EVs[suboptimal])
                    chosen = optimal if np.random.rand() < prob_optimal else suboptimal
                else:
                    pair = transfer_trial_sequence[trial - 150]
                    optimal, suboptimal = (pair[0], pair[1])
                    prob_optimal = self.softmax(self.EVs[optimal], self.EVs[suboptimal])
                    chosen = optimal if np.random.rand() < prob_optimal else suboptimal

                reward = np.random.normal(self.reward_means[chosen], self.reward_sd[chosen])
                trial_details.append(
                    {"trial": trial + 1, "pair": (chr(65 + pair[0]), chr(65 + pair[1])), "choice": chr(65 + chosen),
                     "reward": reward})
                EV_history[trial] = self.update(chosen, reward, trial + 1)

            all_results.append({
                "simulation_num": iteration + 1,
                "trial_indices": trial_indices,
                "t": self.t,
                "a": self.a,
                "b": self.b,
                "trial_details": trial_details,
                "EV_history": EV_history
            })

        return all_results

    def negative_log_likelihood(self, params, reward, choiceset, choice):
        """
        Compute the negative log likelihood for the given parameters and data.

        Parameters:
        - params: Parameters of the model.
        - reward: List or array of observed rewards.
        - choiceset: List or array of available choicesets for each trial.
        - choice: List or array of chosen options for each trial.
        """
        self.reset()

        if self.model_type in ('decay', 'delta', 'decay_choice', 'decay_win'):
            self.t = params[0]
            self.a = params[1]
        elif self.model_type == 'decay_fre':
            self.t = params[0]
            self.a = params[1]
            self.b = params[2]
        elif self.model_type == 'sampler_decay':
            if self.num_params == 2:
                self.t = params[0]
                self.a = params[1]
            elif self.num_params == 3:
                self.t = params[0]
                self.a = params[1]
                self.b = params[2]

        nll = 0

        choiceset_mapping = {
            0: (0, 1),
            1: (2, 3),
            2: (2, 0),
            3: (2, 1),
            4: (0, 3),
            5: (1, 3)
        }

        trial = np.arange(1, self.num_trials + 1)

        for r, cs, ch, trial in zip(reward, choiceset, choice, trial):
            cs_mapped = choiceset_mapping[cs]
            prob_choice = self.softmax(self.EVs[cs_mapped[0]], self.EVs[cs_mapped[1]])
            prob_choice_alt = self.softmax(self.EVs[cs_mapped[1]], self.EVs[cs_mapped[0]])
            nll += -np.log(prob_choice if ch == cs_mapped[0] else prob_choice_alt)
            self.update(ch, r, trial)
        return nll

    def fit(self, data, num_iterations=1, beta_lower=-1, beta_upper=1):
        """
        Fit the model to the provided data.

        Parameters:
        - data: Dictionary of data for each participant.
        - num_iterations: Number of times to repeat the fitting.
        - beta_lower: Lower bound for the beta parameter.
        - beta_upper: Upper bound for the beta parameter.
        """

        all_results = []
        total_nll = 0  # Initialize the cumulative negative log likelihood
        total_n = self.num_trials

        if self.model_type in ('decay', 'delta', 'decay_choice', 'decay_win'):
            k = 2  # Initialize the cumulative number of parameters
        elif self.model_type in ('decay_fre', 'delta_decay'):
            k = 3
        elif self.model_type == 'sampler_decay':
            k = self.num_params

        for participant_id, pdata in data.items():

            print(f"Fitting data for {participant_id}...")
            self.iteration = 0

            best_nll = 100000  # Initialize best negative log likelihood to a large number
            best_initial_guess = None
            best_parameters = None

            for _ in range(num_iterations):  # Randomly initiate the starting parameter for 1000 times

                self.iteration += 1
                print(f"\n=== Iteration {self.iteration} ===\n")

                if self.model_type in ('decay', 'delta', 'decay_choice', 'decay_win'):
                    initial_guess = [np.random.uniform(0, 5), np.random.uniform(0, 1)]
                    bounds = ((0, 5), (0, 1))
                elif self.model_type in ('decay_fre', 'delta_decay'):
                    initial_guess = [np.random.uniform(0, 5), np.random.uniform(0, 1),
                                     np.random.uniform(beta_lower, beta_upper)]
                    bounds = ((0, 5), (0, 1), (beta_lower, beta_upper))
                elif self.model_type == 'sampler_decay':
                    if self.num_params == 2:
                        initial_guess = [np.random.uniform(0, 5), np.random.uniform(0, 1)]
                        bounds = ((0, 5), (0, 1))
                    elif self.num_params == 3:
                        initial_guess = [np.random.uniform(0, 5), np.random.uniform(0, 1),
                                         np.random.uniform(0, 1)]
                        bounds = ((0, 5), (0, 1), (0, 1))

                result = minimize(self.negative_log_likelihood, initial_guess,
                                  args=(pdata['reward'], pdata['choiceset'], pdata['choice']),
                                  bounds=bounds, method='L-BFGS-B', options={'maxiter': 10000})

                if result.fun < best_nll:
                    best_nll = result.fun
                    best_initial_guess = initial_guess
                    best_parameters = result.x

            aic = 2 * k + 2 * best_nll
            bic = k * np.log(total_n) + 2 * best_nll

            total_nll += best_nll

            all_results.append({
                'participant_id': participant_id,
                'best_nll': best_nll,
                'best_initial_guess': best_initial_guess,
                'best_parameters': best_parameters,
                'total_nll': total_nll,
                'AIC': aic,
                'BIC': bic
            })

        return all_results


def likelihood_ratio_test(null_results, alternative_results, df):
    """
    Perform a likelihood ratio test.

    Parameters:
    - null_nll: Negative log-likelihood of the simpler (null) model.
    - alternative_nll: Negative log-likelihood of the more complex (alternative) model.
    - df: Difference in the number of parameters between the two models.

    Returns:
    - p_value: p-value of the test.
    """
    # locate the nll values for the null and alternative models
    null_nll = null_results['best_nll']

    alternative_nll = alternative_results['best_nll']

    # Compute the likelihood ratio statistic
    lr_stat = 2 * np.mean((null_nll - alternative_nll))

    # Get the p-value
    p_value = chi2.sf(lr_stat, df)

    return p_value


def bayes_factor(null_results, alternative_results):
    """
    Compute the Bayes factor.

    Parameters:
    - null_BIC: BIC of the simpler (null) model.
    - alternative_BIC: BIC of the more complex (alternative) model.

    Returns:
    - bayes_factor: Bayes factor of the test.
    """
    # locate the nll values for the null and alternative models
    null_BIC = null_results['BIC']
    alternative_BIC = alternative_results['BIC']

    # Compute the Bayes factor
    log_BF_array = -0.5 * (alternative_BIC - null_BIC)
    mean_log_BF = np.mean(log_BF_array)
    BF = np.exp(mean_log_BF)

    return BF


def dict_generator(df):
    """
    Convert a dataframe into a dictionary.

    Parameters:
    - df: Dataframe to be converted.

    Returns:
    - A dictionary of the dataframe.
    """
    d = {}
    for name, group in df.groupby('Subnum'):
        d[name] = {
            'reward': group['Reward'].tolist(),
            'choiceset': group['SetSeen.'].tolist(),
            'choice': group['KeyResponse'].tolist(),
        }
    return d


def best_param_generator(df, param):
    """

    :param df:
    :param param:
    :return:
    """
    if param == 't':
        t_best = df['best_parameters'].apply(
            lambda x: float(x.strip('[]').split()[0]) if isinstance(x, str) else np.nan
        )
        return t_best

    elif param == 'a':
        a_best = df['best_parameters'].apply(
            lambda x: float(x.strip('[]').split()[1]) if isinstance(x, str) else np.nan
        )
        return a_best

    elif param == 'b':
        b_best = df['best_parameters'].apply(
            lambda x: float(x.strip('[]').split()[2]) if isinstance(x, str) else np.nan
        )
        return b_best



