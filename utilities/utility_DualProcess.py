import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2, dirichlet, multivariate_normal


# This function is used to build and fit a dual-process model
# The idea of the dual-process model is that decision-making, particularly decision-making in the ABCD task,
# is potentially driven by two processes: a Dirichlet process and a Gaussian process.
# When the variance of the underlying reward distribution is small, the Gaussian process (average) dominates the
# decision-making process, whereas when the variance is large, the Dirichlet process (frequency) dominates.


class DualProcessModel:
    def __init__(self, n_samples=1000, num_trials=250, params=False):
        self.num_trials = num_trials
        self.EVs = np.full(4, 0.5)
        self.EV_Dir = np.full(4, 0.5)
        self.EV_Gau = np.full(4, 0.5)
        self.AV = np.full(4, 0.5)
        self.var = np.full(4, 0)
        self.alpha = np.full(4, 1)
        self.n_samples = n_samples
        self.reward_history = [[0] for _ in range(4)]

        self.t = None

    def reset(self):
        self.EV_Dir = np.full(4, 0.5)
        self.EV_Gau = np.full(4, 0.5)
        self.AV = np.full(4, 0.5)
        self.var = np.full(4, 0)
        self.alpha = np.full(4, 1)
        self.reward_history = [[] for _ in range(4)]

    def softmax(self, chosen, alt1):
        c = 3 ** self.t - 1
        num = np.exp(min(700, c * chosen))
        denom = num + np.exp(min(700, c * alt1))
        return num / denom

    def EV_calculation(self, EV_Dir, EV_Gau, weight):
        EVs = weight * EV_Dir + (1 - weight) * EV_Gau
        return EVs

    def update(self, chosen, reward, trial):

        if trial > 150:
            return self.EV_Dir, self.EV_Gau

        else:
            # for every trial, we need to update the EV for both the Dirichlet and Gaussian processes
            # Dirichlet process
            self.alpha[chosen] += 1
            self.EV_Dir = np.mean(dirichlet.rvs(self.alpha, size=self.n_samples), axis=0)

            # Gaussian process
            self.reward_history[chosen].append(reward)
            self.AV = [np.mean(hist) for hist in self.reward_history]  # Calculate mean for each option
            self.var = [np.var(hist) for hist in self.reward_history]  # Calculate variance for each option
            # The four options are independent, so the covariance matrix is diagonal
            cov_matrix = np.diag(self.var)
            self.EV_Gau = np.mean(multivariate_normal.rvs(self.AV, cov_matrix, size=self.n_samples), axis=0)

        return self.EV_Dir, self.EV_Gau

    def unpack_simulation_results(self, results, model):

        unpacked_results = []

        for result in results:
            sim_num = result["simulation_num"]
            t = result["t"]

            for trial_idx, trial_detail, ev_dir, ev_gau in zip(result['trial_indices'],
                                                               result['trial_details'],
                                                               result['EV_history_Dir'],
                                                               result['EV_history_Gau']):

                if model == 'Dir':
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

                elif model == 'Gau':
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

                elif model == 'Dual':
                    var = {
                        "simulation_num": sim_num,
                        "trial_index": trial_idx,
                        "t": t,
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

                elif model == 'Param':
                    var = {
                        "simulation_num": sim_num,
                        "trial_index": trial_idx,
                        "t": t,
                        "pair": trial_detail['pair'],
                        "choice": trial_detail['choice'],
                        "reward": trial_detail['reward'],
                        "weight": trial_detail['weight'],
                        "EV_A": self.EV_calculation(ev_dir[0], ev_gau[0], trial_detail['weight']),
                        "EV_B": self.EV_calculation(ev_dir[1], ev_gau[1], trial_detail['weight']),
                        "EV_C": self.EV_calculation(ev_dir[2], ev_gau[2], trial_detail['weight']),
                        "EV_D": self.EV_calculation(ev_dir[3], ev_gau[3], trial_detail['weight'])
                    }

                unpacked_results.append(var)

            df = pd.DataFrame(unpacked_results)
            return df

    def simulate(self, reward_means, reward_sd, model, AB_freq=None, CD_freq=None,
                 sim_trials=250, num_iterations=1000):

        all_results = []

        for iteration in range(num_iterations):

            print(f"Iteration {iteration + 1} of {num_iterations}")

            self.t = np.random.uniform(0, 5)

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

                if model == 'Dir':
                    prob_optimal = self.softmax(self.EV_Dir[optimal], self.EV_Dir[suboptimal])
                    chosen = optimal if np.random.rand() < prob_optimal else suboptimal

                elif model == 'Gau':
                    prob_optimal = self.softmax(self.EV_Gau[optimal], self.EV_Gau[suboptimal])
                    chosen = optimal if np.random.rand() < prob_optimal else suboptimal

                elif model == 'Dual':
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

                elif model == 'Param':
                    EV_Dir = self.EV_Dir
                    EV_Gau = self.EV_Gau
                    weight = np.random.uniform(0, 1)

                    EVs = self.EV_calculation(EV_Dir, EV_Gau, weight)
                    prob_optimal = self.softmax(EVs[optimal], EVs[suboptimal])
                    chosen = optimal if np.random.rand() < prob_optimal else suboptimal

                reward = np.random.normal(reward_means[chosen], reward_sd[chosen])
                if model == 'Dual':
                    trial_details.append(
                        {"trial": trial + 1, "pair": (chr(65 + pair[0]), chr(65 + pair[1])), "choice": chr(65 + chosen),
                         "reward": reward, "process": process_chosen})
                elif model == 'Param':
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
                "trial_details": trial_details,
                "EV_history_Dir": EV_history_Dir,
                "EV_history_Gau": EV_history_Gau
            })

        return self.unpack_simulation_results(all_results, model)

    def negative_log_likelihood(self, params, reward, choiceset, choice):

        self.reset()

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

        if params is True:
            # Calculate the expected value of the model
            self.EVs = params[0] * self.EV_Dir + (1 - params[0]) * self.EV_Gau

            for r, cs, ch, trial in zip(reward, choiceset, choice, trial):
                cs_mapped = choiceset_mapping[cs]
                prob_choice = self.softmax(self.EVs[cs_mapped[0]], self.EVs[cs_mapped[1]])
                prob_choice_alt = self.softmax(self.EVs[cs_mapped[1]], self.EVs[cs_mapped[0]])
                nll += -np.log(prob_choice if ch == cs_mapped[0] else prob_choice_alt)
                self.update(ch, r, trial)

            return nll

        else:
            # Calculate the nll for two processes individually
            # Choose the process with the lowest nll for each trial

            process_chosen = []

            for r, cs, ch, trial in zip(reward, choiceset, choice, trial):
                cs_mapped = choiceset_mapping[cs]
                dir_prob = self.softmax(self.EV_Dir[cs_mapped[0]], self.EV_Dir[cs_mapped[1]])
                dir_prob_alt = self.softmax(self.EV_Dir[cs_mapped[1]], self.EV_Dir[cs_mapped[0]])
                gau_prob = self.softmax(self.EV_Gau[cs_mapped[0]], self.EV_Gau[cs_mapped[1]])
                gau_prob_alt = self.softmax(self.EV_Gau[cs_mapped[1]], self.EV_Gau[cs_mapped[0]])

                if ch == cs_mapped[0]:
                    process_chosen.append('Dir' if dir_prob > gau_prob else 'Gau')
                    nll += -np.log(dir_prob if dir_prob > gau_prob else gau_prob)
                elif ch == cs_mapped[1]:
                    process_chosen.append('Dir' if dir_prob_alt > gau_prob_alt else 'Gau')
                    nll += -np.log(dir_prob_alt if dir_prob_alt > gau_prob_alt else gau_prob_alt)

                self.update(ch, r, trial)

            return nll, process_chosen






