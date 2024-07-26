import numpy as np
import pandas as pd
import time
from scipy.optimize import minimize
from scipy.stats import dirichlet, multivariate_normal, entropy
from concurrent.futures import ProcessPoolExecutor


# This function is used to build and fit a dual-process model
# The idea of the dual-process model is that decision-making, particularly decision-making in the ABCD task,
# is potentially driven by two processes: a Dirichlet process and a Gaussian process.
# When the variance of the underlying reward distribution is small, the Gaussian process (average) dominates the
# decision-making process, whereas when the variance is large, the Dirichlet process (frequency) dominates.


def fit_participant(model, participant_id, pdata, model_type, weight_fun, num_iterations=1000):
    print(f"Fitting participant {participant_id}...")
    start_time = time.time()

    total_nll = 0
    total_n = model.num_trials

    if model_type in ('Param', 'Recency', 'Param_Dynamic', 'Param_Dynamic', 'Param_Dynamic_Recency', 'Entropy_Recency',
                      'Confidence_Recency', 'Threshold'):
        k = 2
    elif model_type == 'Threshold_Recency':
        k = 3
    elif model_type == 'Multi_Param':
        k = 7
    else:
        k = 1

    if weight_fun == 'pure_weight':
        k = k - 1

    print(k)

    model.iteration = 0

    best_nll = 100000
    best_initial_guess = None
    best_parameters = None

    for _ in range(num_iterations):

        model.iteration += 1

        print('Participant {} - Iteration [{}/{}]'.format(participant_id, model.iteration,
                                                          num_iterations))

        if model_type in ('Dir', 'Gau', 'Param', 'Recency', 'Param_Dynamic', 'Param_Dynamic', 'Param_Dynamic_Recency',
                          'Entropy_Recency', 'Confidence_Recency', 'Threshold'):
            initial_guess = [np.random.uniform(0.0001, 4.9999), np.random.uniform(0.0001, 0.9999)]
            bounds = [(0.0001, 4.9999), (0.0001, 0.9999)]
        elif model_type == 'Threshold_Recency':
            initial_guess = [np.random.uniform(0.0001, 4.9999), np.random.uniform(0.0001, 0.9999),
                             np.random.uniform(0.5001, 0.9999)]
            bounds = [(0.0001, 4.9999), (0.0001, 0.9999), (0.5001, 0.9999)]
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
            best_weight = model.weight_history

    aic = 2 * k + 2 * best_nll
    bic = k * np.log(total_n) + 2 * best_nll

    total_nll += best_nll

    result_dict = {
        'participant_id': participant_id,
        'best_nll': best_nll,
        'best_initial_guess': best_initial_guess,
        'best_parameters': best_parameters,
        'best_process_chosen': best_process_chosen if model_type in ('Dual', 'Recency', 'Threshold',
                                                                     'Threshold_Recency') else None,
        'best_weight': best_weight if model_type in ('Entropy', 'Entropy_Recency', 'Confidence', 'Confidence_Recency',
                                                     'Threshold', 'Threshold_Recency') else None,
        'total_nll': total_nll,
        'AIC': aic,
        'BIC': bic
    }

    print(f"Participant {participant_id} fitted in {(time.time() - start_time) / 60} minutes.")

    return result_dict


def EV_calculation(EV_Dir, EV_Gau, weight):
    EVs = weight * EV_Dir + (1 - weight) * EV_Gau
    return EVs


class DualProcessModel:
    def __init__(self, n_samples=1000, num_trials=250):
        self.iteration = None
        self.num_trials = num_trials
        self.EVs = np.full(4, 0.5)
        self.default_EVs = np.full(4, 0.5)
        self.EV_Dir = np.full(4, 0.5)
        self.EV_Gau = np.full(4, 0.5)
        self.AV = np.full(4, 0.5)
        self.var = np.full(4, 1 / 12)
        self.alpha = np.full(4, 1.0)
        self.gamma_a = np.full(4, 0.5)
        self.gamma_b = np.full(4, 0.0)
        self.n_samples = n_samples
        self.reward_history = [[0] for _ in range(4)]
        self.process_chosen = []
        self.weight_history = []

        self.t = None
        self.a = None
        self.b = None
        self.tau = None
        self.model = None
        self.sim_type = None
        self.arbitration_function = None
        self.Gau_update_fun = None
        self.Dir_update_fun = None
        self.action_selection_Gau = None
        self.action_selection_Dir = None

        self.prior_mean = 0.5
        self.prior_var = 1 / 12

        # Define the mapping between model parameters and input features
        self.choiceset_mapping = {
            0: (0, 1),
            1: (2, 3),
            2: (2, 0),
            3: (2, 1),
            4: (0, 3),
            5: (1, 3)
        }

        self.model_mapping = {
            'Dir': self.dir_nll,
            'Gau': self.gau_nll,
            'Dual': self.dual_nll,
            'Recency': self.recency_nll,
            'Entropy': self.entropy_nll,
            'Entropy_Recency': self.entropy_recency_nll,
            'Confidence': self.confidence_nll,
            'Confidence_Recency': self.confidence_recency_nll,
            'Threshold': self.threshold_nll,
            'Threshold_Recency': self.Threshold_Recency_nll,
            'Param': self.param_nll,
            'Multi_Param': self.multi_param_nll
        }

        self.sim_function_mapping = {
            'Dir': self.single_process_sim,
            'Gau': self.single_process_sim,
            'Dual': self.dual_process_sim,
            'Recency': self.dual_process_sim,
            'Threshold': self.threshold_sim,
            'Threshold_Recency': self.threshold_sim,
            'Param': self.param_sim,
        }

        self.Gau_fun_mapping = {
            'Dir': self.Gau_naive_update,
            'Gau': self.Gau_bayesian_update_with_recency,
            'Dual': self.Gau_bayesian_update,
            'Recency': self.Gau_bayesian_update_with_recency,
            'Entropy': self.Gau_bayesian_update,
            'Entropy_Recency': self.Gau_bayesian_update_with_recency,
            'Confidence': self.Gau_bayesian_update,
            'Confidence_Recency': self.Gau_bayesian_update_with_recency,
            'Threshold': self.Gau_bayesian_update,
            'Threshold_Recency': self.Gau_bayesian_update_with_recency,
            'Param': self.Gau_bayesian_update_with_recency,
            'Multi_Param': self.Gau_bayesian_update_with_recency
        }

        self.Dir_fun_mapping = {
            'Dir': self.Dir_update,
            'Gau': self.Dir_update,
            'Dual': self.Dir_update,
            'Recency': self.Dir_update_with_recency,
            'Entropy': self.Dir_update,
            'Entropy_Recency': self.Dir_update_with_recency,
            'Confidence': self.Dir_update,
            'Confidence_Recency': self.Dir_update_with_recency,
            'Threshold': self.Dir_update,
            'Threshold_Recency': self.Dir_update_with_recency,
            'Param': self.Dir_update_with_recency,
            'Multi_Param': self.Dir_update_with_recency
        }

        self.arbitration_mapping = {
            'Original': self.original_arbitration_mechanism,
            'Max Prob': self.max_prob_arbitration_mechanism
        }

        self.selection_mapping = {
            'softmax': self.softmax,
            'weight': self.weight
        }

    def reset(self):
        self.EV_Dir = np.full(4, 0.5)
        self.EV_Gau = np.full(4, 0.5)
        self.AV = np.full(4, 0.5)
        self.var = np.full(4, 1 / 12)
        self.alpha = np.full(4, 1.0)
        self.gamma_a = np.full(4, 0.5)
        self.gamma_b = np.full(4, 0.0)
        self.reward_history = [[] for _ in range(4)]
        self.process_chosen = []

    def softmax(self, chosen, alt1):
        c = 3 ** self.t - 1
        num = np.exp(min(700, c * chosen))
        denom = num + np.exp(min(700, c * alt1))
        return np.clip(num / denom, 0.0001, 0.9999)

    def weight(self, chosen, alt1):
        weight = chosen / (chosen + alt1)
        return np.clip(weight, 0.0001, 0.9999)

    def original_arbitration_mechanism(self, max_prob, dir_prob, dir_prob_alt, gau_prob, gau_prob_alt, trial_type=None,
                                       chosen=None):
        if chosen == trial_type[0]:
            chosen_process = 'Dir' if dir_prob > gau_prob else 'Gau'
            prob_choice = dir_prob if dir_prob > gau_prob else gau_prob
            prob_choice_alt = 1 - prob_choice
        elif chosen == trial_type[1]:
            chosen_process = 'Dir' if dir_prob_alt > gau_prob_alt else 'Gau'
            prob_choice = dir_prob_alt if dir_prob_alt > gau_prob_alt else gau_prob_alt
            prob_choice_alt = 1 - prob_choice

        return chosen_process, prob_choice, prob_choice_alt

    def max_prob_arbitration_mechanism(self, max_prob, dir_prob, dir_prob_alt, gau_prob, gau_prob_alt, trial_type=None,
                                       chosen=None):
        if max_prob == dir_prob or max_prob == dir_prob_alt:
            chosen_process = 'Dir'
            prob_choice = dir_prob
            prob_choice_alt = dir_prob_alt
        elif max_prob == gau_prob or max_prob == gau_prob_alt:
            chosen_process = 'Gau'
            prob_choice = gau_prob
            prob_choice_alt = gau_prob_alt

        return chosen_process, prob_choice, prob_choice_alt

    def Gau_bayesian_update(self, prior_mean, prior_var, reward, chosen, n=1):
        # since we are conducting sequential Bayesian updating with a batch size of 1, the sample variance needs to be
        # estimated with an inverse gamma distribution

        self.gamma_a[chosen] += n / 2
        self.gamma_b[chosen] += (reward - prior_mean) ** 2 / 2

        if self.gamma_a[chosen] <= 1:
            self.AV[chosen] = reward
            self.var[chosen] = prior_var
        else:
            # sample variance can be directly calculated using a / (b - 1)
            sample_var = self.gamma_b[chosen] / (self.gamma_a[chosen] - 1)

            self.AV[chosen] = (prior_mean * sample_var + reward * n * prior_var) / (prior_var * n + sample_var)
            self.var[chosen] = (prior_var * sample_var) / (n * prior_var + sample_var)

    def Gau_bayesian_update_with_recency(self, prior_mean, prior_var, reward, chosen, n=1):

        self.gamma_a[chosen] += n / 2
        self.gamma_b[chosen] += (reward - prior_mean) ** 2 / 2

        if self.gamma_a[chosen] <= 1:
            self.AV[chosen] = reward
            self.var[chosen] = prior_var
        else:
            # sample variance can be directly calculated using a / (b - 1)
            sample_var = self.gamma_b[chosen] / (self.gamma_a[chosen] - 1)

            self.AV[chosen] = (((1 - self.a) * prior_mean * sample_var + reward * n * prior_var) /
                               (prior_var * n + sample_var * (1 - self.a)))
            self.var[chosen] = (prior_var * sample_var) / (n * prior_var + (1 - self.a) * sample_var)

    def Gau_naive_update(self, prior_mean, prior_var, reward, chosen, n=1):
        self.AV[chosen] = np.mean(self.reward_history[chosen])
        self.var[chosen] = np.var(self.reward_history[chosen])

    def Gau_naive_update_with_recency(self, prior_mean, prior_var, reward, chosen, n=1):
        self.var[chosen] += self.a * ((reward - self.AV[chosen]) ** 2 - self.var[chosen])
        self.AV[chosen] += self.a * (reward - self.AV[chosen])

    def Dir_update(self, chosen, reward, AV_total):
        if reward > AV_total:
            self.alpha[chosen] += 1
        else:
            pass

    def Dir_update_with_recency(self, chosen, reward, AV_total):
        if reward > AV_total:
            self.alpha[chosen] += 1
        else:
            pass

        self.alpha = [np.clip((1 - self.a) * i, 1, 9999) for i in self.alpha]

    def update(self, chosen, reward, trial):

        if trial > 150:
            return self.EV_Dir, self.EV_Gau

        else:
            self.reward_history[chosen].append(reward)

            # for every trial, we need to update the EV for both the Dirichlet and Gaussian processes
            # Dirichlet process
            flatten_reward_history = [item for sublist in self.reward_history for item in sublist]
            AV_total = np.mean(flatten_reward_history)

            self.Gau_update_fun(self.AV[chosen], self.var[chosen], reward, chosen)
            self.Dir_update_fun(chosen, reward, AV_total)

            # Use the updated parameters to get the posterior Dirichlet distribution
            # Sample from the posterior distribution to get the expected value
            self.EV_Dir = np.mean(dirichlet.rvs(self.alpha, size=self.n_samples), axis=0)

            # Otherwise, according to the central limit theorem, the sample mean for n samples randomly drawn from a
            # Dirichlet distribution also follows a Dirichlet distribution with alpha equal to alpha * n

            # # So, we can directly sample from a Dirichlet distribution with alpha * n to get the expected value
            # sample_mean_alpha = [i * self.n_samples for i in self.alpha]
            # self.EV_Dir = dirichlet.rvs(sample_mean_alpha, size=1)[0]

            # Use the updated parameters to get the posterior Gaussian distribution
            # The four options are independent, so the covariance matrix is diagonal
            cov_matrix = np.diag(self.var)

            # Sample from the posterior distribution to get the expected value
            self.EV_Gau = np.mean(multivariate_normal.rvs(self.AV, cov_matrix, size=self.n_samples), axis=0)

            # # Same logic as above
            # self.EV_Gau = multivariate_normal.rvs(self.AV, cov_matrix / self.n_samples, size=1)

        return self.EV_Dir, self.EV_Gau

    # =============================================================================
    # Define the simulation function for each single model
    # This is to reduce the number of if-else statements in the main function and improve time complexity
    # =============================================================================
    def single_process_sim(self, optimal, suboptimal, reward_means, reward_sd, trial_details, pair, trial,
                           weight=None, process=None):
        prob_optimal = self.softmax(getattr(self, process)[optimal], getattr(self, process)[suboptimal])

        chosen = 1 if np.random.rand() < prob_optimal else 0

        reward = np.random.normal(reward_means[optimal if chosen == 1 else suboptimal],
                                  reward_sd[optimal if chosen == 1 else suboptimal])

        trial_details.append(
            {"pair": pair, "choice": chosen, "reward": reward})

        self.update(optimal if chosen == 1 else suboptimal, reward, trial)

        return trial_details

    def param_sim(self, optimal, suboptimal, reward_means, reward_sd, trial_details, pair, trial, weight,
                  process=None):
        EVs = EV_calculation(self.EV_Dir, self.EV_Gau, weight)

        prop_optimal = self.softmax(EVs[optimal], EVs[suboptimal])

        chosen = 1 if np.random.rand() < prop_optimal else 0

        reward = np.random.normal(reward_means[optimal if chosen == 1 else suboptimal],
                                  reward_sd[optimal if chosen == 1 else suboptimal])

        trial_details.append(
            {"pair": pair, "choice": chosen, "reward": reward})

        self.update(optimal if chosen == 1 else suboptimal, reward, trial)

        return trial_details

    def dual_process_sim(self, optimal, suboptimal, reward_means, reward_sd, trial_details, pair, trial,
                         weight=None, process=None):
        prob_optimal_dir = self.softmax(self.EV_Dir[optimal], self.EV_Dir[suboptimal])
        prob_optimal_gau = self.softmax(self.EV_Gau[optimal], self.EV_Gau[suboptimal])
        prob_suboptimal_dir = 1 - prob_optimal_dir
        prob_suboptimal_gau = 1 - prob_optimal_gau

        max_prob = max(prob_optimal_dir, prob_suboptimal_dir, prob_optimal_gau, prob_suboptimal_gau)

        chosen_process, prob_choice, prob_choice_alt = self.max_prob_arbitration_mechanism(max_prob, prob_optimal_dir,
                                                                                           prob_suboptimal_dir,
                                                                                           prob_optimal_gau,
                                                                                           prob_suboptimal_gau,
                                                                                           pair, None)
        chosen = 1 if np.random.rand() < prob_choice else 0

        reward = np.random.normal(reward_means[optimal if chosen == 1 else suboptimal],
                                  reward_sd[optimal if chosen == 1 else suboptimal])

        trial_details.append(
            {"pair": pair, "choice": chosen, "reward": reward, "process": chosen_process})

        self.update(optimal if chosen == 1 else suboptimal, reward, trial)

        return trial_details

    def threshold_sim(self, optimal, suboptimal, reward_means, reward_sd, trial_details, pair, trial, weight=None,
                      process=None):
        prob_optimal_dir = self.softmax(self.EV_Dir[optimal], self.EV_Dir[suboptimal])
        prob_optimal_gau = self.softmax(self.EV_Gau[optimal], self.EV_Gau[suboptimal])
        prob_suboptimal_dir = 1 - prob_optimal_dir
        prob_suboptimal_gau = 1 - prob_optimal_gau

        max_prob = max(prob_optimal_dir, prob_suboptimal_dir, prob_optimal_gau, prob_suboptimal_gau)

        if max_prob > self.tau:
            chosen_process, prob_choice, prob_choice_alt = self.max_prob_arbitration_mechanism(max_prob,
                                                                                               prob_optimal_dir,
                                                                                               prob_suboptimal_dir,
                                                                                               prob_optimal_gau,
                                                                                               prob_suboptimal_gau,
                                                                                               pair, None)
            chosen = 1 if np.random.rand() < prob_choice else 0
        else:
            chosen_process = 'Param'
            EVs = EV_calculation(self.EV_Dir, self.EV_Gau, weight)
            prob_choice = self.softmax(EVs[optimal], EVs[suboptimal])
            chosen = 1 if np.random.rand() < prob_choice else 0

        reward = np.random.normal(reward_means[optimal if chosen == 1 else suboptimal],
                                  reward_sd[optimal if chosen == 1 else suboptimal])

        trial_details.append(
            {"pair": pair, "choice": chosen, "reward": reward, "process": chosen_process})

        self.update(optimal if chosen == 1 else suboptimal, reward, trial)

        return trial_details

    def unpack_simulation_results(self, results):

        unpacked_results = []

        for result in results:
            if 'simulation_num' not in result:  # that means this is a post-hoc simulation
                self.sim_type = 'post-hoc'
                sim_num = result['Subnum']
                t = result["t"]
                a = result["a"] if self.model in ('Recency', 'Threshold_Recency') else None
                weight = result["weight"] if self.model in ('Param', 'Threshold', 'Threshold_Recency') else None
                tau = result["tau"] if self.model in ('Threshold', 'Threshold_Recency') else None

                for trial_idx, trial_detail in zip(result['trial_indices'], result['trial_details']):
                    var = {
                        "Subnum": sim_num,
                        "trial_index": trial_idx,
                        "t": t,
                        "a": a if self.model in ('Recency', 'Threshold_Recency') else None,
                        "weight": weight if self.model in ('Param', 'Threshold', 'Threshold_Recency') else None,
                        "tau": tau if self.model in ('Threshold', 'Threshold_Recency') else None,
                        "pair": trial_detail['pair'],
                        "choice": trial_detail['choice'],
                        "reward": trial_detail['reward'],
                        "process": trial_detail['process'] if self.model in ('Dual', 'Recency', 'Threshold',
                                                                             'Threshold_Recency') else None,
                    }

                    unpacked_results.append(var)

            else:
                self.sim_type = 'a priori'
                sim_num = result["simulation_num"]
                t = result["t"]
                a = result["a"] if self.model in ('Recency', 'Threshold_Recency') else None
                weight = result["weight"] if self.model in ('Param', 'Threshold', 'Threshold_Recency') else None
                tau = result["tau"] if self.model in ('Threshold', 'Threshold_Recency') else None

                for trial_idx, trial_detail, ev_dir, ev_gau in zip(result['trial_indices'],
                                                                   result['trial_details'],
                                                                   result['EV_history_Dir'],
                                                                   result['EV_history_Gau']):
                    var = {
                        "simulation_num": sim_num,
                        "trial_index": trial_idx,
                        "t": t,
                        "a": a if self.model in ('Recency', 'Threshold_Recency') else None,
                        "weight": weight if self.model in ('Param', 'Threshold', 'Threshold_Recency') else None,
                        "tau": tau if self.model in ('Threshold', 'Threshold_Recency') else None,
                        "pair": trial_detail['pair'],
                        "choice": trial_detail['choice'],
                        "reward": trial_detail['reward'],
                        "EV_A_Dir": ev_dir[0] if self.model in ('Dir', 'Dual', 'Recency', 'Threshold',
                                                                'Threshold_Recency') else None,
                        "EV_B_Dir": ev_dir[1] if self.model in ('Dir', 'Dual', 'Recency', 'Threshold',
                                                                'Threshold_Recency') else None,
                        "EV_C_Dir": ev_dir[2] if self.model in ('Dir', 'Dual', 'Recency', 'Threshold',
                                                                'Threshold_Recency') else None,
                        "EV_D_Dir": ev_dir[3] if self.model in ('Dir', 'Dual', 'Recency', 'Threshold',
                                                                'Threshold_Recency') else None,
                        "EV_A_Gau": ev_gau[0] if self.model in ('Gau', 'Dual', 'Recency', 'Threshold',
                                                                'Threshold_Recency') else None,
                        "EV_B_Gau": ev_gau[1] if self.model in ('Gau', 'Dual', 'Recency', 'Threshold',
                                                                'Threshold_Recency') else None,
                        "EV_C_Gau": ev_gau[2] if self.model in ('Gau', 'Dual', 'Recency', 'Threshold',
                                                                'Threshold_Recency') else None,
                        "EV_D_Gau": ev_gau[3] if self.model in ('Gau', 'Dual', 'Recency', 'Threshold',
                                                                'Threshold_Recency') else None,
                        "EV_A": EV_calculation(ev_dir, ev_gau, weight)[0] if self.model in ('Param', 'Threshold',
                                                                                            'Threshold_Recency') else None,
                        "EV_B": EV_calculation(ev_dir, ev_gau, weight)[1] if self.model in ('Param', 'Threshold',
                                                                                            'Threshold_Recency') else None,
                        "EV_C": EV_calculation(ev_dir, ev_gau, weight)[2] if self.model in ('Param', 'Threshold',
                                                                                            'Threshold_Recency') else None,
                        "EV_D": EV_calculation(ev_dir, ev_gau, weight)[3] if self.model in ('Param', 'Threshold',
                                                                                            'Threshold_Recency') else None,
                    }

                    unpacked_results.append(var)

        df = pd.DataFrame(unpacked_results)

        if self.sim_type == 'a priori':
            df = df.dropna(axis=1, how='all')
            return df
        elif self.sim_type == 'post-hoc':
            df['process'] = df['process'].map({'Gau': 0, 'Dir': 1}) if self.model in ('Dual', 'Recency', 'Threshold',
                                                                                      'Threshold_Recency') else None
            df['Param_Process'] = df['process'].isna().astype(int) if self.model in ('Threshold',
                                                                                     'Threshold_Recency') else None
            summary = df.groupby(['Subnum', 'trial_index']).agg(
                pair=('pair', 'first'),
                reward=('reward', 'mean'),
                t=('t', 'mean'),
                a=('a', 'mean'),
                weight=('weight', 'mean'),
                tau=('tau', 'mean'),
                choice=('choice', 'mean'),
                process=('process', 'mean'),
                param_process=('Param_Process', 'mean')
            ).reset_index()

            summary = summary.dropna(axis=1, how='all')

            return summary

    def simulate(self, reward_means, reward_sd, model, AB_freq=None, CD_freq=None,
                 sim_trials=250, num_iterations=1000):

        self.model = model
        self.recency_function = self.recency_mapping[self.model]

        sim_func = self.sim_function_mapping[self.model]

        all_results = []

        for iteration in range(num_iterations):

            print(f"Iteration {iteration + 1} of {num_iterations}")

            self.reset()

            if self.model in ('Dir', 'Gau'):
                process = f'EV_{self.model}'
            else:
                process = None

            # Randomly sample the parameters for the model
            self.t = np.random.uniform(0.0001, 4.9999)
            self.a = np.random.uniform(0.0001, .9999) if self.model in ('Recency', 'Threshold_Recency') else None
            weight = np.random.uniform(0.0001, 0.9999) if self.model in ('Param', 'Threshold',
                                                                         'Threshold_Recency') else None
            self.tau = np.random.uniform(0.5001, 0.9999) if self.model in ('Threshold',
                                                                           'Threshold_Recency') else None

            # Initialize the EVs for the Dirichlet and Gaussian processes
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

                sim_func(optimal, suboptimal, reward_means, reward_sd, trial_details, pair, trial, weight, process)

                EV_history_Dir[trial] = self.EV_Dir
                EV_history_Gau[trial] = self.EV_Gau

            all_results.append({
                "simulation_num": iteration + 1,
                "trial_indices": trial_indices,
                "t": self.t,
                "a": self.a if self.model in ('Recency', 'Threshold_Recency') else None,
                "weight": weight if self.model in ('Param', 'Threshold', 'Threshold_Recency') else None,
                "tau": self.tau if self.model in ('Threshold', 'Threshold_Recency') else None,
                "trial_details": trial_details,
                "EV_history_Dir": EV_history_Dir,
                "EV_history_Gau": EV_history_Gau
            })

        return self.unpack_simulation_results(all_results)

    # =============================================================================
    # Define the negative log likelihood function for each single model
    # This is to reduce the number of if-else statements in the main function and improve time complexity
    # =============================================================================
    """
    Use the following print statement to debug the negative log likelihood functions
    
    # print(f'Trial: {t}, Trial Type: {cs_mapped}, Choice: {ch}, Reward: {r}')
    # print(f'Dir_EV: {self.EV_Dir}')
    # print(f'Gau_EV: {self.EV_Gau}')
    # print(f'Dir_Prob: {dir_prob}, Dir_Prob_Alt: {dir_prob_alt}')
    # print(f'Gau_Prob: {gau_prob}, Gau_Prob_Alt: {gau_prob_alt}')
    # print(f'Dir_Entropy: {dir_entropy}, Gau_Entropy: {gau_entropy}')
    # print(f'Weight: {weight_dir}, Prob_Choice: {prob_choice}')
    
    """

    def param_nll(self, params, reward, choiceset, choice, trial):

        nll = 0

        weight = params[1]

        for r, cs, ch, t in zip(reward, choiceset, choice, trial):

            cs_mapped = self.choiceset_mapping[cs]

            if np.std(self.EV_Dir) == 0 and np.std(self.EV_Gau) == 0:
                self.EVs = self.default_EVs
            else:
                # Standardize the EVs
                EV_Dir = (self.EV_Dir - np.mean(self.EV_Dir)) / np.std(self.EV_Dir)
                EV_Gau = (self.EV_Gau - np.mean(self.EV_Gau)) / np.std(self.EV_Gau)

                # Calculate the expected value as a weighted sum of the two processes
                self.EVs = EV_calculation(EV_Dir, EV_Gau, weight)

            # Calculate the probability of choosing the optimal option
            prob_choice = self.softmax(self.EVs[cs_mapped[0]], self.EVs[cs_mapped[1]])

            nll += -np.log(prob_choice if ch == cs_mapped[0] else 1 - prob_choice)

            # Update the EVs
            self.update(ch, r, t)

        return nll

    def multi_param_nll(self, params, reward, choiceset, choice, trial):

        nll = 0

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
            cs_mapped = self.choiceset_mapping[cs]

            if np.std(self.EV_Dir) == 0 and np.std(self.EV_Gau) == 0:
                self.EVs = self.default_EVs
            else:
                # Standardize the EVs
                EV_Dir = (self.EV_Dir - np.mean(self.EV_Dir)) / np.std(self.EV_Dir)
                EV_Gau = (self.EV_Gau - np.mean(self.EV_Gau)) / np.std(self.EV_Gau)

                weight = weight_mapping[cs]

                # Calculate the expected value of the model
                self.EVs = EV_calculation(EV_Dir, EV_Gau, weight)

            prob_choice = self.softmax(self.EVs[cs_mapped[0]], self.EVs[cs_mapped[1]])
            nll += -np.log(prob_choice if ch == cs_mapped[0] else 1 - prob_choice)

            self.update(ch, r, t)

        return nll

    def dir_nll(self, params, reward, choiceset, choice, trial):

        self.a = params[1]

        nll = 0

        for r, cs, ch, t in zip(reward, choiceset, choice, trial):
            cs_mapped = self.choiceset_mapping[cs]

            prob_choice = self.action_selection_Dir(self.EV_Dir[cs_mapped[0]], self.EV_Dir[cs_mapped[1]])
            nll += -np.log(prob_choice if ch == cs_mapped[0] else 1 - prob_choice)

            self.update(ch, r, t)

        return nll

    def gau_nll(self, params, reward, choiceset, choice, trial):

        self.a = params[1]

        nll = 0

        for r, cs, ch, t in zip(reward, choiceset, choice, trial):
            cs_mapped = self.choiceset_mapping[cs]

            prob_choice = self.action_selection_Gau(self.EV_Gau[cs_mapped[0]], self.EV_Gau[cs_mapped[1]])
            nll += -np.log(prob_choice if ch == cs_mapped[0] else 1 - prob_choice)

            self.update(ch, r, t)

        return nll

    def dual_nll(self, params, reward, choiceset, choice, trial):

        nll = 0

        self.process_chosen = []

        for r, cs, ch, t in zip(reward, choiceset, choice, trial):
            cs_mapped = self.choiceset_mapping[cs]

            dir_prob = self.action_selection_Dir(self.EV_Dir[cs_mapped[0]], self.EV_Dir[cs_mapped[1]])
            dir_prob_alt = 1 - dir_prob
            gau_prob = self.action_selection_Gau(self.EV_Gau[cs_mapped[0]], self.EV_Gau[cs_mapped[1]])
            gau_prob_alt = 1 - gau_prob

            max_prob = max(dir_prob, dir_prob_alt, gau_prob, gau_prob_alt)

            chosen_process, prob_choice, prob_choice_alt = self.arbitration_function(max_prob, dir_prob,
                                                                                     dir_prob_alt,
                                                                                     gau_prob, gau_prob_alt,
                                                                                     cs_mapped, ch)
            self.process_chosen.append(chosen_process)

            nll += -np.log(prob_choice if ch == cs_mapped[0] else prob_choice_alt)

            self.update(ch, r, t)

        return nll

    def recency_nll(self, params, reward, choiceset, choice, trial):

        nll = 0

        self.a = params[1]

        self.process_chosen = []

        for r, cs, ch, t in zip(reward, choiceset, choice, trial):
            cs_mapped = self.choiceset_mapping[cs]

            dir_prob = self.action_selection_Dir(self.EV_Dir[cs_mapped[0]], self.EV_Dir[cs_mapped[1]])
            dir_prob_alt = 1 - dir_prob
            gau_prob = self.action_selection_Gau(self.EV_Gau[cs_mapped[0]], self.EV_Gau[cs_mapped[1]])
            gau_prob_alt = 1 - gau_prob

            max_prob = max(dir_prob, dir_prob_alt, gau_prob, gau_prob_alt)

            chosen_process, prob_choice, prob_choice_alt = self.arbitration_function(max_prob, dir_prob,
                                                                                     dir_prob_alt,
                                                                                     gau_prob, gau_prob_alt,
                                                                                     cs_mapped, ch)
            self.process_chosen.append(chosen_process)

            nll += -np.log(prob_choice if ch == cs_mapped[0] else prob_choice_alt)

            self.update(ch, r, t)

        return nll

    def entropy_nll(self, params, reward, choiceset, choice, trial):

        nll = 0

        self.weight_history = []

        for r, cs, ch, t in zip(reward, choiceset, choice, trial):
            cs_mapped = self.choiceset_mapping[cs]

            dir_prob = self.action_selection_Dir(self.EV_Dir[cs_mapped[0]], self.EV_Dir[cs_mapped[1]])
            dir_prob_alt = 1 - dir_prob
            gau_prob = self.action_selection_Gau(self.EV_Gau[cs_mapped[0]], self.EV_Gau[cs_mapped[1]])
            gau_prob_alt = 1 - gau_prob

            dir_entropy = entropy([dir_prob, dir_prob_alt])
            gau_entropy = entropy([gau_prob, gau_prob_alt])

            weight_dir = gau_entropy / (dir_entropy + gau_entropy)
            self.weight_history.append(weight_dir)

            prob_choice = EV_calculation(dir_prob, gau_prob, weight_dir)

            nll += -np.log(prob_choice if ch == cs_mapped[0] else 1 - prob_choice)

            self.update(ch, r, t)

        return nll

    def entropy_recency_nll(self, params, reward, choiceset, choice, trial):

        nll = 0

        self.weight_history = []

        self.a = params[1]

        for r, cs, ch, t in zip(reward, choiceset, choice, trial):
            cs_mapped = self.choiceset_mapping[cs]

            dir_prob = self.action_selection_Dir(self.EV_Dir[cs_mapped[0]], self.EV_Dir[cs_mapped[1]])
            dir_prob_alt = 1 - dir_prob
            gau_prob = self.action_selection_Gau(self.EV_Gau[cs_mapped[0]], self.EV_Gau[cs_mapped[1]])
            gau_prob_alt = 1 - gau_prob

            dir_entropy = entropy([dir_prob, dir_prob_alt])
            gau_entropy = entropy([gau_prob, gau_prob_alt])

            weight_dir = gau_entropy / (dir_entropy + gau_entropy)
            self.weight_history.append(weight_dir)

            prob_choice = EV_calculation(dir_prob, gau_prob, weight_dir)

            nll += -np.log(prob_choice if ch == cs_mapped[0] else 1 - prob_choice)

            self.update(ch, r, t)

        return nll

    def confidence_nll(self, params, reward, choiceset, choice, trial):

        nll = 0

        self.weight_history = []

        for r, cs, ch, t in zip(reward, choiceset, choice, trial):
            cs_mapped = self.choiceset_mapping[cs]

            dir_prob = self.action_selection_Dir(self.EV_Dir[cs_mapped[0]], self.EV_Dir[cs_mapped[1]])
            dir_prob_alt = 1 - dir_prob
            gau_prob = self.action_selection_Gau(self.EV_Gau[cs_mapped[0]], self.EV_Gau[cs_mapped[1]])
            gau_prob_alt = 1 - gau_prob

            confidence_dir = np.max([dir_prob, dir_prob_alt])
            confidence_gau = np.max([gau_prob, gau_prob_alt])

            weight_dir = confidence_dir / (confidence_dir + confidence_gau)
            self.weight_history.append(weight_dir)

            prob_choice = EV_calculation(dir_prob, gau_prob, weight_dir)

            nll += -np.log(prob_choice if ch == cs_mapped[0] else 1 - prob_choice)

            self.update(ch, r, t)

        return nll

    def confidence_recency_nll(self, params, reward, choiceset, choice, trial):

        nll = 0

        self.a = params[1]

        self.weight_history = []

        for r, cs, ch, t in zip(reward, choiceset, choice, trial):
            cs_mapped = self.choiceset_mapping[cs]

            dir_prob = self.action_selection_Dir(self.EV_Dir[cs_mapped[0]], self.EV_Dir[cs_mapped[1]])
            dir_prob_alt = 1 - dir_prob
            gau_prob = self.action_selection_Gau(self.EV_Gau[cs_mapped[0]], self.EV_Gau[cs_mapped[1]])
            gau_prob_alt = 1 - gau_prob

            confidence_dir = np.max([dir_prob, dir_prob_alt])
            confidence_gau = np.max([gau_prob, gau_prob_alt])

            weight_dir = confidence_dir / (confidence_dir + confidence_gau)
            self.weight_history.append(weight_dir)

            prob_choice = EV_calculation(dir_prob, gau_prob, weight_dir)

            nll += -np.log(prob_choice if ch == cs_mapped[0] else 1 - prob_choice)

            self.update(ch, r, t)

        return nll

    def threshold_nll(self, params, reward, choiceset, choice, trial):

        nll = 0

        self.tau = params[1]

        self.process_chosen = []
        self.weight_history = []

        for r, cs, ch, t in zip(reward, choiceset, choice, trial):
            cs_mapped = self.choiceset_mapping[cs]

            dir_prob = self.action_selection_Dir(self.EV_Dir[cs_mapped[0]], self.EV_Dir[cs_mapped[1]])
            dir_prob_alt = 1 - dir_prob
            gau_prob = self.action_selection_Gau(self.EV_Gau[cs_mapped[0]], self.EV_Gau[cs_mapped[1]])
            gau_prob_alt = 1 - gau_prob

            max_prob = max(dir_prob, dir_prob_alt, gau_prob, gau_prob_alt)

            if max_prob > self.tau:
                chosen_process, prob_choice, prob_choice_alt = self.arbitration_function(max_prob, dir_prob,
                                                                                         dir_prob_alt,
                                                                                         gau_prob,
                                                                                         gau_prob_alt)
                self.process_chosen.append(chosen_process)
                self.weight_history.append(1.0)
            else:
                chosen_process = 'Param'
                self.process_chosen.append(chosen_process)

                dir_entropy = entropy([dir_prob, dir_prob_alt])
                gau_entropy = entropy([gau_prob, gau_prob_alt])

                weight_dir = gau_entropy / (dir_entropy + gau_entropy)
                self.weight_history.append(weight_dir)

                prob_choice = EV_calculation(dir_prob, gau_prob, weight_dir)

            nll += -np.log(prob_choice if ch == cs_mapped[0] else 1 - prob_choice)

            self.update(ch, r, t)

        return nll

    def Threshold_Recency_nll(self, params, reward, choiceset, choice, trial):

        nll = 0

        self.a = params[1]
        self.tau = params[2]

        self.process_chosen = []
        self.weight_history = []

        for r, cs, ch, t in zip(reward, choiceset, choice, trial):
            cs_mapped = self.choiceset_mapping[cs]

            dir_prob = self.softmax(self.EV_Dir[cs_mapped[0]], self.EV_Dir[cs_mapped[1]])
            dir_prob_alt = 1 - dir_prob
            gau_prob = self.softmax(self.EV_Gau[cs_mapped[0]], self.EV_Gau[cs_mapped[1]])
            gau_prob_alt = 1 - gau_prob

            max_prob = max(dir_prob, dir_prob_alt, gau_prob, gau_prob_alt)

            if max_prob > self.tau:
                chosen_process, prob_choice, prob_choice_alt = self.arbitration_function(max_prob, dir_prob,
                                                                                         dir_prob_alt,
                                                                                         gau_prob,
                                                                                         gau_prob_alt)
                self.process_chosen.append(chosen_process)
                self.weight_history.append(1.0)
            else:
                chosen_process = 'Param'
                self.process_chosen.append(chosen_process)

                dir_entropy = entropy([dir_prob, dir_prob_alt])
                gau_entropy = entropy([gau_prob, gau_prob_alt])

                weight_dir = gau_entropy / (dir_entropy + gau_entropy)
                self.weight_history.append(weight_dir)

                prob_choice = EV_calculation(dir_prob, gau_prob, weight_dir)

            nll += -np.log(prob_choice if ch == cs_mapped[0] else 1 - prob_choice)

            self.update(ch, r, t)

        return nll

    def negative_log_likelihood(self, params, reward, choiceset, choice):

        self.reset()

        self.t = params[0]

        trial = np.arange(1, self.num_trials + 1)

        return self.model_mapping[self.model](params, reward, choiceset, choice, trial)

    def fit(self, data, model, num_iterations=100, arbi_option='Max Prob', Gau_fun=None, Dir_fun=None,
            weight_Gau='weight', weight_Dir='weight'):

        self.model = model
        self.arbitration_function = self.arbitration_mapping[arbi_option]

        # Assign the methods based on the provided strings
        self.action_selection_Dir = self.selection_mapping.get(weight_Dir)
        self.action_selection_Gau = self.selection_mapping.get(weight_Gau)

        # Check if both selections are 'weight'
        if self.action_selection_Dir == self.weight and self.action_selection_Gau == self.weight:
            weight_fun = 'pure_weight'
        else:
            weight_fun = 'mixed_weight'

        if Gau_fun is None:
            self.Gau_update_fun = self.Gau_fun_mapping[self.model]
        elif Gau_fun == 'Bayesian':
            self.Gau_update_fun = self.Gau_bayesian_update
        elif Gau_fun == 'Bayesian_Recency':
            self.Gau_update_fun = self.Gau_bayesian_update_with_recency
        elif Gau_fun == 'Naive':
            self.Gau_update_fun = self.Gau_naive_update
        elif Gau_fun == 'Naive_Recency':
            self.Gau_update_fun = self.Gau_naive_update_with_recency

        if Dir_fun is None:
            self.Dir_update_fun = self.Dir_fun_mapping[self.model]
        elif Dir_fun == 'Recency':
            self.Dir_update_fun = self.Dir_update_with_recency

        print(f'============================================================')
        print(f'In the current model, the Dirichlet process is updated using {self.Dir_update_fun.__name__} '
              f'and the Gaussian process is updated using {self.Gau_update_fun.__name__}')
        print(f'Dirichlet process is selected using {self.action_selection_Dir.__name__} and '
              f'Gaussian process is selected using {self.action_selection_Gau.__name__}')
        print(f'============================================================')

        # Creating a list to hold the future results
        futures = []
        results = []

        # Starting a pool of workers with ProcessPoolExecutor
        with ProcessPoolExecutor() as executor:
            # Submitting jobs to the executor for each participant
            for participant_id, participant_data in data.items():
                # fit_participant is the function to be executed in parallel
                future = executor.submit(fit_participant, self, participant_id, participant_data, model, weight_fun,
                                         num_iterations)
                futures.append(future)

            # Collecting results as they complete
            for future in futures:
                results.append(future.result())

        return pd.DataFrame(results)

    def post_hoc_simulation(self, fitting_result, original_data, model, reward_means,
                            reward_sd, num_iterations=1000, Gau_fun=None, Dir_fun=None):

        self.model = model

        if Gau_fun is None:
            self.Gau_update_fun = self.Gau_fun_mapping[self.model]
        elif Gau_fun == 'Bayesian':
            self.Gau_update_fun = self.Gau_bayesian_update
        elif Gau_fun == 'Bayesian_Recency':
            self.Gau_update_fun = self.Gau_bayesian_update_with_recency
        elif Gau_fun == 'Naive':
            self.Gau_update_fun = self.Gau_naive_update
        elif Gau_fun == 'Naive_Recency':
            self.Gau_update_fun = self.Gau_naive_update_with_recency

        if Dir_fun is None:
            self.Dir_update_fun = self.Dir_fun_mapping[self.model]
        elif Dir_fun == 'Recency':
            self.Dir_update_fun = self.Dir_update_with_recency

        post_hoc_func = self.sim_function_mapping[self.model]

        if self.model in ('Dir', 'Gau'):
            process = f'EV_{self.model}'
        else:
            process = None

        # extract the trial sequence for each participant
        trial_index = original_data.groupby('Subnum')['trial_index'].apply(list)
        trial_sequence = original_data.groupby('Subnum')['TrialType'].apply(list)

        t_sequence = fitting_result['best_parameters'].apply(
            lambda x: float(x.strip('[]').split()[0]) if isinstance(x, str) else np.nan)

        if self.model == 'Recency':
            a_sequence = fitting_result['best_parameters'].apply(
                lambda x: float(x.strip('[]').split()[1]) if isinstance(x, str) else np.nan)

        if self.model == 'Param':
            weight_sequence = fitting_result['best_parameters'].apply(
                lambda x: float(x.strip('[]').split()[1]) if isinstance(x, str) else np.nan)

        if self.model == 'Threshold':
            weight_sequence = fitting_result['best_parameters'].apply(
                lambda x: float(x.strip('[]').split()[1]) if isinstance(x, str) else np.nan)
            tau_sequence = fitting_result['best_parameters'].apply(
                lambda x: float(x.strip('[]').split()[2]) if isinstance(x, str) else np.nan)

        if self.model == 'Threshold_Recency':
            weight_sequence = fitting_result['best_parameters'].apply(
                lambda x: float(x.strip('[]').split()[1]) if isinstance(x, str) else np.nan)
            a_sequence = fitting_result['best_parameters'].apply(
                lambda x: float(x.strip('[]').split()[2]) if isinstance(x, str) else np.nan)
            tau_sequence = fitting_result['best_parameters'].apply(
                lambda x: float(x.strip('[]').split()[3]) if isinstance(x, str) else np.nan)

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
            start_time = time.time()

            self.t = t_sequence[participant - 1]
            weight = weight_sequence[participant - 1] if self.model in (
                'Param', 'Threshold', 'Threshold_Recency') else None
            self.a = a_sequence[participant - 1] if self.model in ('Recency', 'Threshold_Recency') else None
            self.tau = tau_sequence[participant - 1] if self.model in ('Threshold', 'Threshold_Recency') else None

            for _ in range(num_iterations):

                print(f"Iteration {_ + 1} of {num_iterations}")

                self.reset()

                self.iteration = 0

                trial_details = []
                trial_indices = []

                for trial, pair in zip(trial_index[participant], trial_sequence[participant]):
                    trial_indices.append(trial)

                    optimal, suboptimal = choice_set_mapping[pair]

                    post_hoc_func(optimal, suboptimal, reward_means, reward_sd, trial_details, pair, trial, weight,
                                  process)

                all_results.append({
                    "Subnum": participant,
                    "t": self.t,
                    "a": self.a if self.model in ('Recency', 'Threshold_Recency') else None,
                    "weight": weight if self.model in ('Param', 'Threshold', 'Threshold_Recency') else None,
                    "tau": self.tau if self.model in ('Threshold', 'Threshold_Recency') else None,
                    "trial_indices": trial_indices,
                    "trial_details": trial_details
                })

            print(f"Post-hoc simulation for participant {participant} finished in {(time.time() - start_time) / 60} "
                  f"minutes")

        return self.unpack_simulation_results(all_results)
