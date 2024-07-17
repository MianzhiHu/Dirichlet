import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2, dirichlet, multivariate_normal
from concurrent.futures import ProcessPoolExecutor


# This function is used to build and fit a dual-process model
# The idea of the dual-process model is that decision-making, particularly decision-making in the ABCD task,
# is potentially driven by two processes: a Dirichlet process and a Gaussian process.
# When the variance of the underlying reward distribution is small, the Gaussian process (average) dominates the
# decision-making process, whereas when the variance is large, the Dirichlet process (frequency) dominates.


def fit_participant(model, participant_id, pdata, model_type, num_iterations=1000):
    print(f"Fitting participant {participant_id}...")

    total_nll = 0
    total_n = model.num_trials

    if model_type in ('Param', 'Recency'):
        k = 2
    elif model_type == 'Multi_Param':
        k = 7
    else:
        k = 1

    model.iteration = 0

    best_nll = 100000
    best_initial_guess = None
    best_parameters = None

    for _ in range(num_iterations):

        model.iteration += 1

        print('Participant {} - Iteration [{}/{}]'.format(participant_id, model.iteration,
                                                          num_iterations))

        if model_type in ('Param', 'Recency'):
            initial_guess = [np.random.uniform(0.0001, 4.9999), np.random.uniform(0.0001, 0.9999)]
            bounds = [(0.0001, 4.9999), (0.0001, 0.9999)]
        elif model_type == 'Multi_Param':
            initial_guess = [np.random.uniform(0.0001, 4.9999), np.random.uniform(0.0001, 0.9999),
                             np.random.uniform(0.0001, 0.9999), np.random.uniform(0.0001, 0.9999),
                             np.random.uniform(0.0001, 0.9999), np.random.uniform(0.0001, 0.9999),
                             np.random.uniform(0.0001, 0.9999)]
            bounds = [(0.0001, 4.9999), (0.0001, 0.9999), (0.0001, 0.9999), (0.0001, 0.9999),
                      (0.0001, 0.9999), (0.0001, 0.9999), (0.0001, 0.9999)]
        else:
            initial_guess = [np.random.uniform(0.0001, 4.9999)]
            bounds = [(0.0001, 4.9999)]

        result = minimize(model.negative_log_likelihood, initial_guess,
                          args=(pdata['reward'], pdata['choiceset'], pdata['choice']),
                          bounds=bounds, method='L-BFGS-B', options={'maxiter': 10000})

        if result.fun < best_nll:
            best_nll = result.fun
            best_initial_guess = initial_guess
            best_parameters = result.x
            best_process_chosen = model.process_chosen

    aic = 2 * k + 2 * best_nll
    bic = k * np.log(total_n) + 2 * best_nll

    total_nll += best_nll

    result_dict = {
        'participant_id': participant_id,
        'best_nll': best_nll,
        'best_initial_guess': best_initial_guess,
        'best_parameters': best_parameters,
        'best_process_chosen': best_process_chosen,
        'total_nll': total_nll,
        'AIC': aic,
        'BIC': bic
    }

    return result_dict


def EV_calculation(EV_Dir, EV_Gau, weight):
    EVs = weight * EV_Dir + (1 - weight) * EV_Gau
    return EVs


class DualProcessModel:
    def __init__(self, n_samples=1000, num_trials=250):
        self.iteration = None
        self.num_trials = num_trials
        self.EVs = np.full(4, 0.5)
        self.EV_Dir = np.full(4, 0.5)
        self.EV_Gau = np.full(4, 0.5)
        self.AV = np.full(4, 0.5)
        self.var = np.full(4, 0.0)
        self.alpha = np.full(4, 1.0)
        self.n_samples = n_samples
        self.reward_history = [[0] for _ in range(4)]
        self.process_chosen = []

        self.t = None
        self.a = None
        self.b = None
        self.model = None
        self.sim_type = None

    def reset(self):
        self.EV_Dir = np.full(4, 0.5)
        self.EV_Gau = np.full(4, 0.5)
        self.AV = np.full(4, 0.5)
        self.var = np.full(4, 0.0)
        self.alpha = np.full(4, 1.0)
        self.reward_history = [[] for _ in range(4)]
        self.process_chosen = []

    def softmax(self, chosen, alt1):
        c = 3 ** self.t - 1
        num = np.exp(min(700, c * chosen))
        denom = num + np.exp(min(700, c * alt1))
        return num / denom

    def update(self, chosen, reward, trial):

        if trial > 150:
            return self.EV_Dir, self.EV_Gau

        else:
            self.reward_history[chosen].append(reward)

            # for every trial, we need to update the EV for both the Dirichlet and Gaussian processes
            # Dirichlet process
            flatten_reward_history = [item for sublist in self.reward_history for item in sublist]
            AV_total = np.mean(flatten_reward_history)

            if self.model == 'Recency':
                self.alpha = [max(np.finfo(float).tiny, (1 - self.a) * i) for i in self.alpha]  # avoid alpha = 0

            if reward > AV_total:
                self.alpha[chosen] += 1
            else:
                pass

            self.EV_Dir = np.mean(dirichlet.rvs(self.alpha, size=self.n_samples), axis=0)

            # Gaussian process
            if self.model == 'Recency':
                self.var[chosen] = self.var[chosen] + self.a * ((reward - self.AV[chosen]) ** 2 - self.var[chosen])
                self.AV[chosen] += self.a * (reward - self.AV[chosen])

            else:
                self.AV = [np.mean(hist) if len(hist) > 0 else 0.5 for hist in self.reward_history]
                self.var = [np.var(hist) if len(hist) > 0 else 0 for hist in self.reward_history]

            # The four options are independent, so the covariance matrix is diagonal
            cov_matrix = np.diag(self.var)

            self.EV_Gau = np.mean(multivariate_normal.rvs(self.AV, cov_matrix, size=self.n_samples), axis=0)

        return self.EV_Dir, self.EV_Gau

    def unpack_simulation_results(self, results):

        unpacked_results = []

        for result in results:
            if 'simulation_num' not in result:  # that means this is a post-hoc simulation
                self.sim_type = 'post-hoc'
                sim_num = result['Subnum']
                t = result["t"]

                for trial_idx, trial_detail in zip(result['trial_indices'], result['trial_details']):
                    var = {
                        "Subnum": sim_num,
                        "trial_index": trial_idx,
                        "t": t,
                        "pair": trial_detail['pair'],
                        "choice": trial_detail['choice'],
                        "reward": trial_detail['reward'],
                        "process": trial_detail['process'] if self.model == 'Dual' else None,
                        "weight": trial_detail['weight'] if self.model == 'Param' else None,
                    }

                    unpacked_results.append(var)

            else:
                self.sim_type = 'a priori'
                sim_num = result["simulation_num"]
                t = result["t"]

                if self.model == 'Recency':
                    a = result["a"]

                for trial_idx, trial_detail, ev_dir, ev_gau in zip(result['trial_indices'],
                                                                   result['trial_details'],
                                                                   result['EV_history_Dir'],
                                                                   result['EV_history_Gau']):

                    if self.model == 'Dir':
                        var = {
                            "simulation_num": sim_num,
                            "trial_index": trial_idx,
                            "t": t,
                            "pair": trial_detail['pair'],
                            "choice": trial_detail['choice'],
                            "reward": trial_detail['reward'],
                            "EV_A": ev_dir[0],
                            "EV_B": ev_dir[1],
                            "EV_C": ev_dir[2],
                            "EV_D": ev_dir[3]
                        }

                    elif self.model == 'Gau':
                        var = {
                            "simulation_num": sim_num,
                            "trial_index": trial_idx,
                            "t": t,
                            "pair": trial_detail['pair'],
                            "choice": trial_detail['choice'],
                            "reward": trial_detail['reward'],
                            "EV_A": ev_gau[0],
                            "EV_B": ev_gau[1],
                            "EV_C": ev_gau[2],
                            "EV_D": ev_gau[3]
                        }

                    elif self.model in ('Dual', 'Recency'):
                        var = {
                            "simulation_num": sim_num,
                            "trial_index": trial_idx,
                            "t": t,
                            "a": a if self.model == 'Recency' else None,
                            "pair": trial_detail['pair'],
                            "choice": trial_detail['choice'],
                            "reward": trial_detail['reward'],
                            "process": trial_detail['process'],
                            "EV_A_Dir": ev_dir[0],
                            "EV_B_Dir": ev_dir[1],
                            "EV_C_Dir": ev_dir[2],
                            "EV_D_Dir": ev_dir[3],
                            "EV_A_Gau": ev_gau[0],
                            "EV_B_Gau": ev_gau[1],
                            "EV_C_Gau": ev_gau[2],
                            "EV_D_Gau": ev_gau[3]
                        }

                    elif self.model == 'Param':
                        var = {
                            "simulation_num": sim_num,
                            "trial_index": trial_idx,
                            "t": t,
                            "pair": trial_detail['pair'],
                            "choice": trial_detail['choice'],
                            "reward": trial_detail['reward'],
                            "weight": trial_detail['weight'],
                            "EV_A": EV_calculation(ev_dir[0], ev_gau[0], trial_detail['weight']),
                            "EV_B": EV_calculation(ev_dir[1], ev_gau[1], trial_detail['weight']),
                            "EV_C": EV_calculation(ev_dir[2], ev_gau[2], trial_detail['weight']),
                            "EV_D": EV_calculation(ev_dir[3], ev_gau[3], trial_detail['weight'])
                        }

                    unpacked_results.append(var)

        df = pd.DataFrame(unpacked_results)

        if self.sim_type == 'a priori':
            return df
        elif self.sim_type == 'post-hoc':
            df['process'] = df['process'].map({'Gau': 0, 'Dir': 1}) if self.model == 'Dual' else None
            summary = df.groupby(['Subnum', 'trial_index'])[
                ['t', 'reward', 'choice', 'process']].mean().reset_index()
            return summary

    def simulate(self, reward_means, reward_sd, model, AB_freq=None, CD_freq=None,
                 sim_trials=250, num_iterations=1000):

        self.model = model
        all_results = []

        for iteration in range(num_iterations):

            print(f"Iteration {iteration + 1} of {num_iterations}")

            self.reset()

            self.t = np.random.uniform(0.0001, 4.9999)

            if self.model == 'Recency':
                self.a = np.random.uniform(0.0001, .9999)

            EV_history_Dir = np.zeros((sim_trials, 4))
            EV_history_Gau = np.zeros((sim_trials, 4))
            trial_details = []
            trial_indices = []

            training_trials = [(0, 1), (2, 3)]
            training_trial_sequence = [training_trials[0]] * AB_freq + [training_trials[1]] * CD_freq
            np.random.shuffle(training_trial_sequence)

            # Distributing the next 100 trials equally among the four pairs (AC, AD, BC, BD)
            transfer_trials = [(2, 0), (1, 3), (0, 3), (2, 1)]
            transfer_trial_sequence = transfer_trials * 25
            np.random.shuffle(transfer_trial_sequence)

            for trial in range(sim_trials):
                trial_indices.append(trial + 1)

                if trial < 150:
                    pair = training_trial_sequence[trial]
                else:
                    pair = transfer_trial_sequence[trial - 150]

                optimal, suboptimal = (pair[0], pair[1])

                if self.model == 'Dir':
                    prob_optimal = self.softmax(self.EV_Dir[optimal], self.EV_Dir[suboptimal])
                    chosen = optimal if np.random.rand() < prob_optimal else suboptimal

                elif self.model == 'Gau':
                    prob_optimal = self.softmax(self.EV_Gau[optimal], self.EV_Gau[suboptimal])
                    chosen = optimal if np.random.rand() < prob_optimal else suboptimal

                elif self.model in ('Dual', 'Recency'):
                    prob_optimal_dir = self.softmax(self.EV_Dir[optimal], self.EV_Dir[suboptimal])
                    prob_optimal_gau = self.softmax(self.EV_Gau[optimal], self.EV_Gau[suboptimal])
                    prob_suboptimal_dir = self.softmax(self.EV_Dir[suboptimal], self.EV_Dir[optimal])
                    prob_suboptimal_gau = self.softmax(self.EV_Gau[suboptimal], self.EV_Gau[optimal])

                    chosen_dir = optimal if np.random.rand() < prob_optimal_dir else suboptimal
                    chosen_gau = optimal if np.random.rand() < prob_optimal_gau else suboptimal

                    max_prob = max(prob_optimal_dir, prob_suboptimal_dir, prob_optimal_gau, prob_suboptimal_gau)

                    if max_prob == prob_optimal_dir or max_prob == prob_suboptimal_dir:
                        process_chosen = 'Dir'
                        chosen = chosen_dir
                    elif max_prob == prob_optimal_gau or max_prob == prob_suboptimal_gau:
                        process_chosen = 'Gau'
                        chosen = chosen_gau

                elif self.model == 'Param':
                    EV_Dir = self.EV_Dir
                    EV_Gau = self.EV_Gau
                    weight = np.random.uniform(0, 1)

                    EVs = EV_calculation(EV_Dir, EV_Gau, weight)
                    prob_optimal = self.softmax(EVs[optimal], EVs[suboptimal])
                    chosen = optimal if np.random.rand() < prob_optimal else suboptimal

                reward = np.random.normal(reward_means[chosen], reward_sd[chosen])

                if self.model in ('Dual', 'Recency'):
                    trial_details.append(
                        {"trial": trial + 1, "pair": (chr(65 + pair[0]), chr(65 + pair[1])), "choice": chr(65 + chosen),
                         "reward": reward, "process": process_chosen})
                elif self.model == 'Param':
                    trial_details.append(
                        {"trial": trial + 1, "pair": (chr(65 + pair[0]), chr(65 + pair[1])), "choice": chr(65 + chosen),
                         "reward": reward, "weight": weight})
                else:
                    trial_details.append(
                        {"trial": trial + 1, "pair": (chr(65 + pair[0]), chr(65 + pair[1])), "choice": chr(65 + chosen),
                         "reward": reward})
                EV_history_Dir[trial], EV_history_Gau[trial] = self.update(chosen, reward, trial)

            all_results.append({
                "simulation_num": iteration + 1,
                "trial_indices": trial_indices,
                "t": self.t,
                "a": self.a if self.model == 'Recency' else None,
                "trial_details": trial_details,
                "EV_history_Dir": EV_history_Dir,
                "EV_history_Gau": EV_history_Gau
            })

        return self.unpack_simulation_results(all_results)

    def negative_log_likelihood(self, params, reward, choiceset, choice):

        self.reset()

        self.t = params[0]

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

        # the most complicated model
        if self.model == 'Multi_Param':

            # Decide which trial type it is
            weight_mapping = {
                0: params[1],
                1: params[2],
                2: params[3],
                3: params[4],
                4: params[5],
                5: params[6]
            }

            for r, cs, ch, t in zip(reward, choiceset, choice, trial):
                # Standardize the EVs
                EV_Dir = (self.EV_Dir - np.mean(self.EV_Dir)) / np.std(self.EV_Dir)
                EV_Gau = (self.EV_Gau - np.mean(self.EV_Gau)) / np.std(self.EV_Gau)

                weight = weight_mapping[cs]

                # Calculate the expected value of the model
                self.EVs = weight * EV_Dir + (1 - weight) * EV_Gau

                cs_mapped = choiceset_mapping[cs]
                prob_choice = self.softmax(self.EVs[cs_mapped[0]], self.EVs[cs_mapped[1]])
                prob_choice_alt = self.softmax(self.EVs[cs_mapped[1]], self.EVs[cs_mapped[0]])

                nll += -np.log(prob_choice if ch == cs_mapped[0] else prob_choice_alt)
                self.update(ch, r, t)

        if self.model == 'Param':

            for r, cs, ch, trial in zip(reward, choiceset, choice, trial):
                # Standardize the EVs
                EV_Dir = (self.EV_Dir - np.mean(self.EV_Dir)) / np.std(self.EV_Dir)
                EV_Gau = (self.EV_Gau - np.mean(self.EV_Gau)) / np.std(self.EV_Gau)

                # Calculate the expected value of the model
                self.EVs = params[1] * EV_Dir + (1 - params[1]) * EV_Gau

                cs_mapped = choiceset_mapping[cs]
                prob_choice = self.softmax(self.EVs[cs_mapped[0]], self.EVs[cs_mapped[1]])
                prob_choice_alt = self.softmax(self.EVs[cs_mapped[1]], self.EVs[cs_mapped[0]])
                nll += -np.log(prob_choice if ch == cs_mapped[0] else prob_choice_alt)
                self.update(ch, r, trial)

        elif self.model == 'Dir':
            for r, cs, ch, trial in zip(reward, choiceset, choice, trial):
                cs_mapped = choiceset_mapping[cs]
                prob_choice = self.softmax(self.EV_Dir[cs_mapped[0]], self.EV_Dir[cs_mapped[1]])
                prob_choice_alt = self.softmax(self.EV_Dir[cs_mapped[1]], self.EV_Dir[cs_mapped[0]])
                nll += -np.log(prob_choice if ch == cs_mapped[0] else prob_choice_alt)
                self.update(ch, r, trial)

        elif self.model == 'Gau':
            for r, cs, ch, trial in zip(reward, choiceset, choice, trial):
                cs_mapped = choiceset_mapping[cs]
                prob_choice = self.softmax(self.EV_Gau[cs_mapped[0]], self.EV_Gau[cs_mapped[1]])
                prob_choice_alt = self.softmax(self.EV_Gau[cs_mapped[1]], self.EV_Gau[cs_mapped[0]])
                nll += -np.log(prob_choice if ch == cs_mapped[0] else prob_choice_alt)
                self.update(ch, r, trial)

        elif self.model in ('Dual', 'Recency'):
            # Calculate the nll for two processes individually
            # Choose the process with the lowest nll for each trial

            self.process_chosen = []

            if self.model == 'Recency':
                self.a = params[1]

            for r, cs, ch, trial in zip(reward, choiceset, choice, trial):
                cs_mapped = choiceset_mapping[cs]

                dir_prob = self.softmax(self.EV_Dir[cs_mapped[0]], self.EV_Dir[cs_mapped[1]])
                dir_prob_alt = self.softmax(self.EV_Dir[cs_mapped[1]], self.EV_Dir[cs_mapped[0]])
                gau_prob = self.softmax(self.EV_Gau[cs_mapped[0]], self.EV_Gau[cs_mapped[1]])
                gau_prob_alt = self.softmax(self.EV_Gau[cs_mapped[1]], self.EV_Gau[cs_mapped[0]])

                max_prob = max(dir_prob, dir_prob_alt, gau_prob, gau_prob_alt)

                if max_prob == dir_prob or max_prob == dir_prob_alt:
                    chosen_process = 'Dir'
                    self.process_chosen.append(chosen_process)
                    prob_choice = dir_prob
                    prob_choice_alt = dir_prob_alt
                elif max_prob == gau_prob or max_prob == gau_prob_alt:
                    chosen_process = 'Gau'
                    self.process_chosen.append(chosen_process)
                    prob_choice = gau_prob
                    prob_choice_alt = gau_prob_alt

                nll += -np.log(prob_choice if ch == cs_mapped[0] else prob_choice_alt)

                self.update(ch, r, trial)

        return nll

    def fit(self, data, model, num_iterations=1000):

        self.model = model

        # Creating a list to hold the future results
        futures = []
        results = []

        # Starting a pool of workers with ProcessPoolExecutor
        with ProcessPoolExecutor() as executor:
            # Submitting jobs to the executor for each participant
            for participant_id, participant_data in data.items():
                # fit_participant is the function to be executed in parallel
                future = executor.submit(fit_participant, self, participant_id, participant_data, model, num_iterations)
                futures.append(future)

            # Collecting results as they complete
            for future in futures:
                results.append(future.result())

        return pd.DataFrame(results)

    def post_hoc_simulation(self, fitting_result, original_data, model, reward_means,
                            reward_sd, num_iterations=1000):

        self.model = model

        t_sequence = fitting_result['best_parameters'].apply(
            lambda x: float(x.strip('[]').split()[0]) if isinstance(x, str) else np.nan)

        if self.model == 'Recency':
            a_sequence = fitting_result['best_parameters'].apply(
                lambda x: float(x.strip('[]').split()[1]) if isinstance(x, str) else np.nan)

        # extract the trial sequence for each participant
        trial_sequence = original_data.groupby('Subnum')['TrialType'].apply(list)
        trial_index = original_data.groupby('Subnum')['trial_index'].apply(list)

        # create a mapping of the choice set to the pair of options
        choice_set_mapping = {
            'AB': (0, 1),
            'CD': (2, 3),
            'CA': (2, 0),
            'CB': (2, 1),
            'BD': (0, 3),
            'AD': (1, 3)
        }

        # start the simulation
        all_results = []

        for participant in fitting_result['participant_id']:
            print(f"Participant {participant}")

            for _ in range(num_iterations):

                print(f"Iteration {_ + 1} of {num_iterations}")

                self.reset()

                self.t = t_sequence[participant - 1]
                self.a = a_sequence[participant - 1] if self.model == 'Recency' else None
                self.iteration = 0

                trial_details = []
                trial_indices = []

                for trial, pair in zip(trial_index[participant], trial_sequence[participant]):

                    trial_indices.append(trial)

                    optimal, suboptimal = choice_set_mapping[pair]

                    if self.model == 'Dir':
                        prob_optimal = self.softmax(self.EV_Dir[optimal], self.EV_Dir[suboptimal])
                        chosen = 1 if np.random.rand() < prob_optimal else 0

                    elif self.model == 'Gau':
                        prob_optimal = self.softmax(self.EV_Gau[optimal], self.EV_Gau[suboptimal])
                        chosen = 1 if np.random.rand() < prob_optimal else 0

                    elif self.model in ('Dual', 'Recency'):
                        prob_optimal_dir = self.softmax(self.EV_Dir[optimal], self.EV_Dir[suboptimal])
                        prob_optimal_gau = self.softmax(self.EV_Gau[optimal], self.EV_Gau[suboptimal])
                        prob_suboptimal_dir = self.softmax(self.EV_Dir[suboptimal], self.EV_Dir[optimal])
                        prob_suboptimal_gau = self.softmax(self.EV_Gau[suboptimal], self.EV_Gau[optimal])

                        chosen_dir = 1 if np.random.rand() < prob_optimal_dir else 0
                        chosen_gau = 1 if np.random.rand() < prob_optimal_gau else 0

                        max_prob = max(prob_optimal_dir, prob_suboptimal_dir, prob_optimal_gau, prob_suboptimal_gau)

                        if max_prob == prob_optimal_dir or max_prob == prob_suboptimal_dir:
                            process_chosen = 'Dir'
                            chosen = chosen_dir
                        elif max_prob == prob_optimal_gau or max_prob == prob_suboptimal_gau:
                            process_chosen = 'Gau'
                            chosen = chosen_gau

                    elif self.model == 'Param':
                        EV_Dir = self.EV_Dir
                        EV_Gau = self.EV_Gau
                        weight = np.random.uniform(0, 1)

                        EVs = EV_calculation(EV_Dir, EV_Gau, weight)
                        prob_optimal = self.softmax(EVs[optimal], EVs[suboptimal])
                        chosen = 1 if np.random.rand() < prob_optimal else 0

                    reward = np.random.normal(reward_means[optimal if chosen == 1 else suboptimal],
                                              reward_sd[optimal if chosen == 1 else suboptimal])

                    if self.model in ('Dual', 'Recency'):
                        trial_details.append(
                            {"pair": pair, "choice": chosen, "reward": reward, "process": process_chosen})
                    elif self.model == 'Param':
                        trial_details.append(
                            {"pair": pair, "choice": chosen, "reward": reward, "weight": weight})
                    else:
                        trial_details.append(
                            {"pair": pair, "choice": chosen, "reward": reward})

                    self.update(optimal if chosen == 1 else suboptimal, reward, trial)

                all_results.append({
                    "Subnum": participant,
                    "t": self.t,
                    "trial_indices": trial_indices,
                    "trial_details": trial_details
                })

        return self.unpack_simulation_results(all_results)
