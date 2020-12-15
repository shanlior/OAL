"""
Reference: https://github.com/openai/imitation
I follow the architecture from the official repository
"""
import gym
import tensorflow as tf
import numpy as np
from stable_baselines.common.running_mean_std import RunningMeanStd, RunningMinMax
from stable_baselines.common import tf_util as tf_util



def logsigmoid(input_tensor):
    """
    Equivalent to tf.log(tf.sigmoid(a))

    :param input_tensor: (tf.Tensor)
    :return: (tf.Tensor)
    """
    return -tf.nn.softplus(-input_tensor)


def logit_bernoulli_entropy(logits):
    """
    Reference:
    https://github.com/openai/imitation/blob/99fbccf3e060b6e6c739bdf209758620fcdefd3c/policyopt/thutil.py#L48-L51

    :param logits: (tf.Tensor) the logits
    :return: (tf.Tensor) the Bernoulli entropy
    """
    ent = (1. - tf.nn.sigmoid(logits)) * logits - logsigmoid(logits)
    return ent
#
class TabularAdversary(object):
    def __init__(self, observation_space, action_space, hidden_size,
                 entcoeff=0.00, scope="adversary", normalize=True, expert_features=None,
                 exploration_bonus=False, bonus_coef=0.01, t_c=0.1):
        """
        Reward regression from observations and transitions

        :param observation_space: (gym.spaces)
        :param action_space: (gym.spaces)
        :param hidden_size: ([int]) the hidden dimension for the MLP
        :param entcoeff: (float) the entropy loss weight
        :param scope: (str) tensorflow variable scope
        :param normalize: (bool) Whether to normalize the reward or not
        """
        # TODO: support images properly (using a CNN)
        self.scope = scope
        self.observation_shape = observation_space.shape
        self.actions_shape = action_space.shape
        if isinstance(action_space, gym.spaces.Box):
            # Continuous action space
            self.discrete_actions = False
            self.n_actions = action_space.shape[0]
        elif isinstance(action_space, gym.spaces.Discrete):
            self.n_actions = action_space.n
            self.discrete_actions = True
        else:
            raise ValueError('Action space not supported: {}'.format(action_space))

        self.hidden_size = hidden_size
        self.normalize = normalize
        self.obs_rms = None
        self.expert_features = expert_features
        self.reward = expert_features
        normalization = np.linalg.norm(self.reward)
        self.norm_factor = np.sqrt(float(self.observation_shape[0]))
        if normalization > 1:
            self.reward = self.reward / (self.norm_factor * normalization)
        self.exploration_bonus = exploration_bonus
        self.t_c = t_c

        self.bonus_coef = bonus_coef
        if self.exploration_bonus:
            self.covariance_lambda = np.identity(self.observation_shape[0])
        else:
            self.covariance_lambda = None


    def update_reward(self, features):

        t_c = self.t_c
        # self.reward = (1-t_c) * self.reward + t_c * (self.expert_features - features)
        self.reward = self.reward + t_c * (self.expert_features - features) / self.norm_factor
        normalization = np.linalg.norm(self.reward)

        if normalization > 1:
            self.reward = self.reward / normalization

    def get_reward(self, observation):
        """
        Predict the reward using the observation and action

        :param obs: (tf.Tensor or np.ndarray) the observation
        :param actions: (tf.Tensor or np.ndarray) the action
        :return: (np.ndarray) the reward
        """
        if self.exploration_bonus:
            self.covariance_lambda = self.covariance_lambda \
                                + np.matmul(np.expand_dims(observation, axis=1), np.expand_dims(observation, axis=0))
            inverse_covariance = np.linalg.inv(self.covariance_lambda)
            reward = np.matmul(observation, np.reshape(self.reward, (self.reward.shape[0], 1))).squeeze()
            bonus = np.sqrt(np.matmul(np.matmul(observation,inverse_covariance), observation))
            return reward + self.bonus_coef * bonus
        else:
            reward = np.matmul(observation, np.reshape(self.reward, (self.reward.shape[0], 1))).squeeze()
            return reward



class TabularAdversaryTF(object):
    def __init__(self, sess, observation_space, action_space, hidden_size,
                 entcoeff=0.00, scope="adversary", normalize=True, expert_features=None,
                 exploration_bonus=False, is_action_features=True, bonus_coef=0.01, t_c=0.1):
        """
        Reward regression from observations and transitions

        :param observation_space: (gym.spaces)
        :param action_space: (gym.spaces)
        :param hidden_size: ([int]) the hidden dimension for the MLP
        :param entcoeff: (float) the entropy loss weight
        :param scope: (str) tensorflow variable scope
        :param normalize: (bool) Whether to normalize the reward or not
        """
        # TODO: support images properly (using a CNN)
        self.scope = scope
        self.sess = sess
        self.observation_shape = observation_space.shape
        self.actions_shape = action_space.shape
        self.is_action_features = is_action_features

        if isinstance(action_space, gym.spaces.Box):
            # Continuous action space
            self.discrete_actions = False
            self.n_actions = action_space.shape[0]
        elif isinstance(action_space, gym.spaces.Discrete):
            self.n_actions = action_space.n
            self.discrete_actions = True
        else:
            raise ValueError('Action space not supported: {}'.format(action_space))

        if self.is_action_features:
            self.n_features = self.observation_shape[0] + self.n_actions
        else:
            self.n_features = self.observation_shape[0]
        expert_features = expert_features[:self.n_features]


        self.hidden_size = hidden_size
        self.normalize = normalize
        self.obs_rms = None

        self.expert_features = tf.constant(expert_features, dtype=tf.float32)
        self.norm_factor = tf.sqrt(float(self.n_features))


        # self.normalization = tf.square(float(np.linalg.norm(expert_features)))
        self.normalization = np.linalg.norm(expert_features)

        expert_normalization = np.linalg.norm(expert_features)


        # if expert_normalization > 1:
        # self.reward_vec = tf.Variable(expert_features / (self.norm_factor * expert_normalization), dtype=tf.float32)
        self.reward_vec = tf.Variable(expert_features, dtype=tf.float32)

        # else:
        #     self.reward_vec = tf.Variable(expert_features / self.norm_factor)
            # self.reward_vec = tf.Variable(expert_features, dtype=tf.float32)

        # self.reward_vec = tf.Variable(expert_features / self.normalization, dtype=tf.float32)

        # if normalization > 1:
        #     self.reward_vec = tf.Variable(expert_features / normalization, dtype=tf.float32)
        # else:
        #     self.reward_vec = tf.Variable(expert_features, dtype=tf.float32)
        #

        self.exploration_bonus = exploration_bonus
        self.t_c = t_c

        self.bonus_coef = bonus_coef
        if self.exploration_bonus:
            self.covariance_lambda = tf.Variable(tf.eye(self.n_features), dtype=tf.float32)
            self.inverse_covariance = tf.eye(self.n_features)
        else:
            self.covariance_lambda = None
            self.inverse_covariance = None
        # Placeholders
        self.features_ph = tf.placeholder(tf.float32, (None,) + (self.n_features, ),
                                               name="observations_ph")
        self.successor_features_ph = tf.placeholder(tf.float32, (self.n_features, ),
                                               name="successor_features_ph")
        # Build graph
        with tf.variable_scope(self.scope, reuse=False):
            if self.normalize:
                with tf.variable_scope("obfilter"):
                    self.obs_rms = RunningMeanStd(shape=self.n_features)
                    # self.obs_rms = RunningMinMax(shape=self.observation_shape)
                obs_scaled = (tf.cast(self.features_ph, tf.float32) - self.obs_rms.mean)\
                      / tf.cast(tf.sqrt(self.obs_rms.var), tf.float32)
                reward_vec_scaled = (tf.cast(self.reward_vec, tf.float32) - self.obs_rms.mean)\
                             / (tf.cast(tf.sqrt(self.obs_rms.var), tf.float32))
                # obs_scaled = (tf.cast(self.features_ph, tf.float32)) / tf.cast(self.obs_rms.scale, tf.float32)
                obs = obs_scaled


                # reward_vec_scaled = (tf.cast(self.reward_vec, tf.float32)) / tf.cast(self.obs_rms.scale,
                #                                                                              tf.float32)
                reward_vec = reward_vec_scaled / tf.norm(reward_vec_scaled)

            else:
                obs = self.features_ph
                reward_vec = self.reward_vec

            if self.exploration_bonus:
                self.new_covariance_lambda = self.covariance_lambda \
                                         + tf.reduce_sum(
                                            tf.matmul(tf.expand_dims(tf.cast(self.features_ph, tf.float32), axis=2),
                                                         tf.expand_dims(tf.cast(self.features_ph, tf.float32), axis=1)), axis=0)
                self.update_covariance_op = tf.assign(self.covariance_lambda, self.new_covariance_lambda)

                bonus = tf.squeeze(tf.sqrt(tf.matmul(tf.matmul(tf.expand_dims(obs, axis=1), self.inverse_covariance),
                                          tf.expand_dims(obs, axis=2))))

                reward = tf.squeeze(tf.matmul(obs, tf.expand_dims(reward_vec, 1)))
                self.reward_op = reward + self.bonus_coef * bonus
            else:
                self.reward_op = tf.squeeze(tf.matmul(obs, tf.expand_dims(reward_vec, 1)))

            # Update reward
            self.new_reward_vec = self.reward_vec + self.t_c * (self.expert_features - self.successor_features_ph)
            # self.new_reward_vec = self.reward_vec\
            #                       + self.t_c * (self.expert_features - self.successor_features_ph)\
            #                       / (self.normalization * self.norm_factor)
            # normalization = tf.norm(self.new_reward_vec) * self.normalization
            # normalization = tf.norm(self.new_reward_vec)
            # self.new_reward_vec = tf.cond(normalization > 1.0,
            #                                 true_fn=lambda: self.new_reward_vec / normalization,
            #                                 false_fn=lambda: self.new_reward_vec)
            # self.new_reward_vec = self.new_reward_vec / normalization

            # reward_vec_unnormalized = self.reward_vec + self.t_c * (self.expert_features - self.successor_features_ph)
            # reward_vec_scaled = (tf.cast(reward_vec_unnormalized, tf.float32)) / tf.cast(self.obs_rms.scale, tf.float32)
            # self.new_reward_vec = reward_vec_scaled / tf.norm(reward_vec_scaled)
            self.update_reward_op = tf.assign(self.reward_vec, self.new_reward_vec)



    def update_reward(self, successor_features):
        #
        # sess = tf.get_default_session()
        # if len(features.shape) == 1:
            # features = np.expand_dims(features, 0)
        if not self.is_action_features:
            successor_features = successor_features[:self.observation_shape[0]]

        feed_dict = {self.successor_features_ph: successor_features}

        if self.exploration_bonus:
            self.inverse_covariance = tf.linalg.inv(self.covariance_lambda)

        self.sess.run(self.update_reward_op, feed_dict)




    def get_reward(self, obs, action=None):
        """
        Predict the reward using the observation and action

        :param obs: (tf.Tensor or np.ndarray) the observation
        :param actions: (tf.Tensor or np.ndarray) the action
        :return: (np.ndarray) the reward
        """
        # sess = tf.get_default_session()
        if len(obs.shape) == 1:
            obs = np.expand_dims(obs, 0)
        if len(action.shape) == 1:
            action = np.expand_dims(action, 0)

        if self.is_action_features:
            features = np.concatenate((obs, action), axis=1)
        else:
            features = obs

        feed_dict = {self.features_ph: features}

        if self.exploration_bonus:
            reward, _ = self.sess.run([self.reward_op, self.update_covariance_op], feed_dict)
        else:
            reward = self.sess.run(self.reward_op, feed_dict)
        return reward



class NeuralAdversary(object):
    def __init__(self, sess, observation_space, action_space, hidden_size=64, scope="adversary", normalize=True):
        """
        Reward regression from observations and transitions

        :param observation_space: (gym.spaces)
        :param action_space: (gym.spaces)
        :param hidden_size: ([int]) the hidden dimension for the MLP
        :param entcoeff: (float) the entropy loss weight
        :param scope: (str) tensorflow variable scope
        :param normalize: (bool) Whether to normalize the reward or not
        """
        # TODO: support images properly (using a CNN)
        self.sess = sess
        self.scope = scope
        self.observation_shape = observation_space.shape
        self.actions_shape = action_space.shape

        if isinstance(action_space, gym.spaces.Box):
            # Continuous action space
            self.discrete_actions = False
            self.n_actions = action_space.shape[0]
        elif isinstance(action_space, gym.spaces.Discrete):
            self.n_actions = action_space.n
            self.discrete_actions = True
        else:
            raise ValueError('Action space not supported: {}'.format(action_space))

        self.hidden_size = hidden_size
        self.normalize = normalize
        self.obs_rms = None

        # Placeholders
        self.generator_obs_ph = tf.placeholder(observation_space.dtype, (None,) + self.observation_shape,
                                               name="observations_ph")
        self.generator_acs_ph = tf.placeholder(action_space.dtype, (None,) + self.actions_shape,
                                               name="actions_ph")
        self.generator_gammas_ph = tf.placeholder(tf.float32, (None, 1), name="gammas_ph")
        self.expert_obs_ph = tf.placeholder(observation_space.dtype, (None,) + self.observation_shape,
                                            name="expert_observations_ph")
        self.expert_acs_ph = tf.placeholder(action_space.dtype, (None,) + self.actions_shape,
                                            name="expert_actions_ph")
        self.expert_gammas_ph = tf.placeholder(tf.float32, (None, 1), name="gammas_ph")

        # Build graph
        generator_rewards = self.build_graph(self.generator_obs_ph, self.generator_acs_ph, reuse=False)
        expert_rewards = self.build_graph(self.expert_obs_ph, self.expert_acs_ph, reuse=True)

        generator_scaled_rewards = tf.multiply(generator_rewards, self.generator_gammas_ph)
        # generator_scaled_rewards = generator_rewards
        generator_value = tf.reduce_sum(generator_scaled_rewards)
        expert_scaled_rewards = tf.multiply(expert_rewards, self.expert_gammas_ph)
        # expert_scaled_rewards = expert_rewards
        expert_value = tf.reduce_sum(expert_scaled_rewards)

        loss = generator_value - expert_value

        # Loss + Accuracy terms
        self.losses = [loss]
        self.loss_name = ["generator_loss", "expert_loss", "entropy", "entropy_loss", "generator_acc", "expert_acc"]
        # self.total_loss = loss
        # Build Reward for policy
        self.reward_op = tf.stop_gradient(generator_rewards)

        var_list = self.get_trainable_variables()
        rewards_optimizer = tf.train.AdamOptimizer(learning_rate=3e-4)
        grads, vars = zip(*rewards_optimizer.compute_gradients(loss, var_list=var_list))
        # grads, vars = list(zip(*grads_and_vars))
        # grads, norm = tf.clip_by_global_norm(grads, 300.0)
        rewards_train_op = rewards_optimizer.apply_gradients(zip(grads, vars))
        norm = tf.constant(0.)
        # rewards_train_op = rewards_optimizer.minimize(loss, var_list=var_list)

        rewards_train_op = [rewards_train_op, norm]

        self.train = tf_util.function(
            [self.generator_obs_ph, self.generator_acs_ph, self.generator_gammas_ph,
             self.expert_obs_ph, self.expert_acs_ph, self.expert_gammas_ph], rewards_train_op)
        # self.lossandgrad = tf_util.function(
        #     [self.generator_obs_ph, self.generator_acs_ph, self.generator_gammas_ph,
        #      self.expert_obs_ph, self.expert_acs_ph, self.expert_gammas_ph],
        #     self.losses + [tf_util.flatgrad(self.total_loss, var_list)])


    def build_graph(self, obs_ph, acs_ph, reuse=False):
        """
        build the graph

        :param obs_ph: (tf.Tensor) the observation placeholder
        :param acs_ph: (tf.Tensor) the action placeholder
        :param reuse: (bool)
        :return: (tf.Tensor) the graph output
        """
        with tf.variable_scope(self.scope):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            if self.normalize:
                with tf.variable_scope("obfilter"):
                    self.obs_rms = RunningMeanStd(shape=self.observation_shape)
                obs = (tf.cast(obs_ph, tf.float32) - self.obs_rms.mean) / tf.cast(tf.sqrt(self.obs_rms.var), tf.float32)
            else:
                obs = obs_ph

            if self.discrete_actions:
                one_hot_actions = tf.one_hot(acs_ph, self.n_actions)
                actions_ph = tf.cast(one_hot_actions, tf.float32)
            else:
                actions_ph = acs_ph

            _input = tf.concat([obs, actions_ph], axis=1)  # concatenate the two input -> form a transition
            p_h1 = tf.contrib.layers.fully_connected(_input, self.hidden_size, activation_fn=tf.nn.tanh)
            p_h2 = tf.contrib.layers.fully_connected(p_h1, self.hidden_size, activation_fn=tf.nn.tanh)
            # rewards = tf.contrib.layers.fully_connected(p_h2, 1, activation_fn=tf.nn.tanh)
            rewards = tf.contrib.layers.fully_connected(p_h2, 1, activation_fn=tf.math.sigmoid)
        return rewards

    def get_trainable_variables(self):
        """
        Get all the trainable variables from the graph

        :return: ([tf.Tensor]) the variables
        """
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def get_reward(self, obs, actions):
        """
        Predict the reward using the observation and action

        :param obs: (tf.Tensor or np.ndarray) the observation
        :param actions: (tf.Tensor or np.ndarray) the action
        :return: (np.ndarray) the reward
        """
        # sess = tf.get_default_session()
        if len(obs.shape) == 1:
            obs = np.expand_dims(obs, 0)
        if len(actions.shape) == 1:
            actions = np.expand_dims(actions, 0)
        elif len(actions.shape) == 0:
            # one discrete action
            actions = np.expand_dims(actions, 0)

        feed_dict = {self.generator_obs_ph: obs, self.generator_acs_ph: actions}
        reward = self.sess.run(self.reward_op, feed_dict)
        return reward


# class TabularAdversaryTFBackup(object):
#     def __init__(self, sess, observation_space, action_space, hidden_size,
#                  entcoeff=0.00, scope="adversary", normalize=True, expert_features=None,
#                  exploration_bonus=False, is_action_features=True, bonus_coef=0.01, t_c=0.1):
#         """
#         Reward regression from observations and transitions
#
#         :param observation_space: (gym.spaces)
#         :param action_space: (gym.spaces)
#         :param hidden_size: ([int]) the hidden dimension for the MLP
#         :param entcoeff: (float) the entropy loss weight
#         :param scope: (str) tensorflow variable scope
#         :param normalize: (bool) Whether to normalize the reward or not
#         """
#         # TODO: support images properly (using a CNN)
#         self.scope = scope
#         self.sess = sess
#         self.observation_shape = observation_space.shape
#         self.actions_shape = action_space.shape
#         self.is_action_features = is_action_features
#
#         if isinstance(action_space, gym.spaces.Box):
#             # Continuous action space
#             self.discrete_actions = False
#             self.n_actions = action_space.shape[0]
#         elif isinstance(action_space, gym.spaces.Discrete):
#             self.n_actions = action_space.n
#             self.discrete_actions = True
#         else:
#             raise ValueError('Action space not supported: {}'.format(action_space))
#
#         if self.is_action_features:
#             self.n_features = self.observation_shape[0] + self.n_actions
#         else:
#             self.n_features = self.observation_shape[0]
#         expert_features = expert_features[:self.n_features]
#
#
#         self.hidden_size = hidden_size
#         self.normalize = normalize
#         self.obs_rms = None
#
#         self.expert_features = tf.constant(expert_features, dtype=tf.float32)
#         self.norm_factor = tf.sqrt(float(self.n_features))
#
#
#         # self.normalization = tf.square(float(np.linalg.norm(expert_features)))
#         self.normalization = np.linalg.norm(expert_features)
#
#         expert_normalization = np.linalg.norm(expert_features)
#
#
#         # if expert_normalization > 1:
#         # self.reward_vec = tf.Variable(expert_features / (self.norm_factor * expert_normalization), dtype=tf.float32)
#         self.reward_vec = tf.Variable(expert_features, dtype=tf.float32)
#
#         # else:
#         #     self.reward_vec = tf.Variable(expert_features / self.norm_factor)
#             # self.reward_vec = tf.Variable(expert_features, dtype=tf.float32)
#
#         # self.reward_vec = tf.Variable(expert_features / self.normalization, dtype=tf.float32)
#
#         # if normalization > 1:
#         #     self.reward_vec = tf.Variable(expert_features / normalization, dtype=tf.float32)
#         # else:
#         #     self.reward_vec = tf.Variable(expert_features, dtype=tf.float32)
#         #
#
#         self.exploration_bonus = exploration_bonus
#         self.t_c = t_c
#
#         self.bonus_coef = bonus_coef
#         if self.exploration_bonus:
#             self.covariance_lambda = tf.Variable(tf.eye(self.n_features), dtype=tf.float32)
#             self.inverse_covariance = tf.eye(self.n_features)
#         else:
#             self.covariance_lambda = None
#             self.inverse_covariance = None
#         # Placeholders
#         self.features_ph = tf.placeholder(tf.float32, (None,) + (self.n_features, ),
#                                                name="observations_ph")
#         self.successor_features_ph = tf.placeholder(tf.float32, (self.n_features, ),
#                                                name="successor_features_ph")
#         # Build graph
#         with tf.variable_scope(self.scope, reuse=False):
#             if self.normalize:
#                 with tf.variable_scope("obfilter"):
#                     self.obs_rms = RunningMeanStd(shape=self.n_features)
#                     # self.obs_rms = RunningMinMax(shape=self.observation_shape)
#                 obs_scaled = (tf.cast(self.features_ph, tf.float32) - self.obs_rms.mean)\
#                       / tf.cast(tf.sqrt(self.obs_rms.var), tf.float32)
#                 reward_vec_scaled = (tf.cast(self.reward_vec, tf.float32) - self.obs_rms.mean)\
#                              / (tf.cast(tf.sqrt(self.obs_rms.var), tf.float32))
#                 # obs_scaled = (tf.cast(self.features_ph, tf.float32)) / tf.cast(self.obs_rms.scale, tf.float32)
#                 obs = obs_scaled
#
#
#                 # reward_vec_scaled = (tf.cast(self.reward_vec, tf.float32)) / tf.cast(self.obs_rms.scale,
#                 #                                                                              tf.float32)
#                 reward_vec = reward_vec_scaled / tf.norm(reward_vec_scaled)
#
#             else:
#                 obs = self.features_ph
#                 reward_vec = self.reward_vec
#
#             if self.exploration_bonus:
#                 self.new_covariance_lambda = self.covariance_lambda \
#                                          + tf.reduce_sum(
#                                             tf.matmul(tf.expand_dims(tf.cast(self.features_ph, tf.float32), axis=2),
#                                                          tf.expand_dims(tf.cast(self.features_ph, tf.float32), axis=1)), axis=0)
#                 self.update_covariance_op = tf.assign(self.covariance_lambda, self.new_covariance_lambda)
#
#                 bonus = tf.squeeze(tf.sqrt(tf.matmul(tf.matmul(tf.expand_dims(obs, axis=1), self.inverse_covariance),
#                                           tf.expand_dims(obs, axis=2))))
#
#                 reward = tf.squeeze(tf.matmul(obs, tf.expand_dims(reward_vec, 1)))
#                 self.reward_op = reward + self.bonus_coef * bonus
#             else:
#                 self.reward_op = tf.squeeze(tf.matmul(obs, tf.expand_dims(reward_vec, 1)))
#
#             # Update reward
#             self.new_reward_vec = self.reward_vec + self.t_c * (self.expert_features - self.successor_features_ph)
#             # self.new_reward_vec = self.reward_vec\
#             #                       + self.t_c * (self.expert_features - self.successor_features_ph)\
#             #                       / (self.normalization * self.norm_factor)
#             # normalization = tf.norm(self.new_reward_vec) * self.normalization
#             # normalization = tf.norm(self.new_reward_vec)
#             # self.new_reward_vec = tf.cond(normalization > 1.0,
#             #                                 true_fn=lambda: self.new_reward_vec / normalization,
#             #                                 false_fn=lambda: self.new_reward_vec)
#             # self.new_reward_vec = self.new_reward_vec / normalization
#
#             # reward_vec_unnormalized = self.reward_vec + self.t_c * (self.expert_features - self.successor_features_ph)
#             # reward_vec_scaled = (tf.cast(reward_vec_unnormalized, tf.float32)) / tf.cast(self.obs_rms.scale, tf.float32)
#             # self.new_reward_vec = reward_vec_scaled / tf.norm(reward_vec_scaled)
#             self.update_reward_op = tf.assign(self.reward_vec, self.new_reward_vec)
#
#
#
#     def update_reward(self, successor_features):
#         #
#         # sess = tf.get_default_session()
#         # if len(features.shape) == 1:
#             # features = np.expand_dims(features, 0)
#         if not self.is_action_features:
#             successor_features = successor_features[:self.observation_shape[0]]
#
#         feed_dict = {self.successor_features_ph: successor_features}
#
#         if self.exploration_bonus:
#             self.inverse_covariance = tf.linalg.inv(self.covariance_lambda)
#
#         self.sess.run(self.update_reward_op, feed_dict)
#
#
#
#
#     def get_reward(self, obs, action=None):
#         """
#         Predict the reward using the observation and action
#
#         :param obs: (tf.Tensor or np.ndarray) the observation
#         :param actions: (tf.Tensor or np.ndarray) the action
#         :return: (np.ndarray) the reward
#         """
#         # sess = tf.get_default_session()
#         if len(obs.shape) == 1:
#             obs = np.expand_dims(obs, 0)
#         if len(action.shape) == 1:
#             action = np.expand_dims(action, 0)
#
#         if self.is_action_features:
#             features = np.concatenate((obs, action), axis=1)
#         else:
#             features = obs
#
#         feed_dict = {self.features_ph: features}
#
#         if self.exploration_bonus:
#             reward, _ = self.sess.run([self.reward_op, self.update_covariance_op], feed_dict)
#         else:
#             reward = self.sess.run(self.reward_op, feed_dict)
#         return reward