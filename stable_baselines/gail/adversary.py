"""
Reference: https://github.com/openai/imitation
I follow the architecture from the official repository
"""
import gym
import tensorflow as tf
import numpy as np

# from stable_baselines.common.mpi_running_mean_std import RunningMeanStd
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

# class TabularAdversary(object):
#     def __init__(self, observation_space, action_space, hidden_size,
#                  entcoeff=0.001, scope="adversary", normalize=True, expert_features=None,
#                  exploration_bonus=False, bonus_coef=0.01, t_c=0.1):
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
#         self.observation_shape = observation_space.shape
#         self.actions_shape = action_space.shape
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
#         self.hidden_size = hidden_size
#         self.normalize = normalize
#         self.obs_rms = None
#         self.expert_features = expert_features
#         self.reward = expert_features
#         normalization = np.linalg.norm(self.reward)
#         self.norm_factor = np.sqrt(float(self.observation_shape[0]))
#         if normalization > 1:
#             self.reward = self.reward / (self.norm_factor * normalization)
#         self.exploration_bonus = exploration_bonus
#         self.t_c = t_c
#
#         self.bonus_coef = bonus_coef
#         if self.exploration_bonus:
#             self.covariance_lambda = np.identity(self.observation_shape[0])
#         else:
#             self.covariance_lambda = None
#
#
#     def update_reward(self, features):
#
#         t_c = self.t_c
#         # self.reward = (1-t_c) * self.reward + t_c * (self.expert_features - features)
#         self.reward = self.reward + t_c * (self.expert_features - features) / self.norm_factor
#         normalization = np.linalg.norm(self.reward)
#
#         if normalization > 1:
#             self.reward = self.reward / normalization
#
#     def get_reward(self, observation):
#         """
#         Predict the reward using the observation and action
#
#         :param obs: (tf.Tensor or np.ndarray) the observation
#         :param actions: (tf.Tensor or np.ndarray) the action
#         :return: (np.ndarray) the reward
#         """
#         if self.exploration_bonus:
#             self.covariance_lambda = self.covariance_lambda \
#                                 + np.matmul(np.expand_dims(observation, axis=1), np.expand_dims(observation, axis=0))
#             inverse_covariance = np.linalg.inv(self.covariance_lambda)
#             reward = np.matmul(observation, np.reshape(self.reward, (self.reward.shape[0], 1))).squeeze()
#             bonus = np.sqrt(np.matmul(np.matmul(observation,inverse_covariance), observation))
#             return reward + self.bonus_coef * bonus
#         else:
#             reward = np.matmul(observation, np.reshape(self.reward, (self.reward.shape[0], 1))).squeeze()
#             return reward

class TransitionClassifier(object):
    def __init__(self, observation_space, action_space, hidden_size,
                 entcoeff=0.001, scope="adversary", normalize=True):
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

        # Placeholders
        self.generator_obs_ph = tf.placeholder(observation_space.dtype, (None,) + self.observation_shape,
                                               name="observations_ph")
        self.generator_acs_ph = tf.placeholder(action_space.dtype, (None,) + self.actions_shape,
                                               name="actions_ph")
        self.expert_obs_ph = tf.placeholder(observation_space.dtype, (None,) + self.observation_shape,
                                            name="expert_observations_ph")
        self.expert_acs_ph = tf.placeholder(action_space.dtype, (None,) + self.actions_shape,
                                            name="expert_actions_ph")
        # Build graph
        generator_logits = self.build_graph(self.generator_obs_ph, self.generator_acs_ph, reuse=False)
        expert_logits = self.build_graph(self.expert_obs_ph, self.expert_acs_ph, reuse=True)
        # Build accuracy
        generator_acc = tf.reduce_mean(tf.cast(tf.nn.sigmoid(generator_logits) < 0.5, tf.float32))
        expert_acc = tf.reduce_mean(tf.cast(tf.nn.sigmoid(expert_logits) > 0.5, tf.float32))
        # Build regression loss
        # let x = logits, z = targets.
        # z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
        generator_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=generator_logits,
                                                                 labels=tf.zeros_like(generator_logits))
        generator_loss = tf.reduce_mean(generator_loss)
        expert_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=expert_logits, labels=tf.ones_like(expert_logits))
        expert_loss = tf.reduce_mean(expert_loss)
        # Build entropy loss
        logits = tf.concat([generator_logits, expert_logits], 0)
        entropy = tf.reduce_mean(logit_bernoulli_entropy(logits))
        entropy_loss = -entcoeff * entropy
        # Loss + Accuracy terms
        self.losses = [generator_loss, expert_loss, entropy, entropy_loss, generator_acc, expert_acc]
        self.loss_name = ["generator_loss", "expert_loss", "entropy", "entropy_loss", "generator_acc", "expert_acc"]
        self.total_loss = generator_loss + expert_loss + entropy_loss
        # Build Reward for policy
        self.reward_op = -tf.log(1 - tf.nn.sigmoid(generator_logits) + 1e-8)
        var_list = self.get_trainable_variables()
        self.lossandgrad = tf_util.function(
            [self.generator_obs_ph, self.generator_acs_ph, self.expert_obs_ph, self.expert_acs_ph],
            self.losses + [tf_util.flatgrad(self.total_loss, var_list)])

    class SuccessorClassifier(object):
        def __init__(self, observation_space, action_space, hidden_size,
                     entcoeff=0.001, scope="adversary", normalize=True):
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

            # Placeholders
            self.generator_obs_ph = tf.placeholder(observation_space.dtype, (None,) + self.observation_shape,
                                                   name="observations_ph")
            self.generator_acs_ph = tf.placeholder(action_space.dtype, (None,) + self.actions_shape,
                                                   name="actions_ph")
            self.expert_obs_ph = tf.placeholder(observation_space.dtype, (None,) + self.observation_shape,
                                                name="expert_observations_ph")
            self.expert_acs_ph = tf.placeholder(action_space.dtype, (None,) + self.actions_shape,
                                                name="expert_actions_ph")
            # Build graph
            generator_logits = self.build_graph(self.generator_obs_ph, self.generator_acs_ph, reuse=False)
            expert_logits = self.build_graph(self.expert_obs_ph, self.expert_acs_ph, reuse=True)
            # Build accuracy
            generator_acc = tf.reduce_mean(tf.cast(tf.nn.sigmoid(generator_logits) < 0.5, tf.float32))
            expert_acc = tf.reduce_mean(tf.cast(tf.nn.sigmoid(expert_logits) > 0.5, tf.float32))
            # Build regression loss
            # let x = logits, z = targets.
            # z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
            generator_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=generator_logits,
                                                                     labels=tf.zeros_like(generator_logits))
            generator_loss = tf.reduce_mean(generator_loss)
            expert_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=expert_logits,
                                                                  labels=tf.ones_like(expert_logits))
            expert_loss = tf.reduce_mean(expert_loss)
            # Build entropy loss
            logits = tf.concat([generator_logits, expert_logits], 0)
            entropy = tf.reduce_mean(logit_bernoulli_entropy(logits))
            entropy_loss = -entcoeff * entropy
            # Loss + Accuracy terms
            self.losses = [generator_loss, expert_loss, entropy, entropy_loss, generator_acc, expert_acc]
            self.loss_name = ["generator_loss", "expert_loss", "entropy", "entropy_loss", "generator_acc", "expert_acc"]
            self.total_loss = generator_loss + expert_loss + entropy_loss
            # Build Reward for policy
            self.reward_op = -tf.log(1 - tf.nn.sigmoid(generator_logits) + 1e-8)
            var_list = self.get_trainable_variables()
            self.lossandgrad = tf_util.function(
                [self.generator_obs_ph, self.generator_acs_ph, self.expert_obs_ph, self.expert_acs_ph],
                self.losses + [tf_util.flatgrad(self.total_loss, var_list)])

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
                obs = (tf.cast(obs_ph, tf.float32) - self.obs_rms.mean) / self.obs_rms.std
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
            logits = tf.contrib.layers.fully_connected(p_h2, 1, activation_fn=tf.identity)
        return logits

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
        sess = tf.get_default_session()
        if len(obs.shape) == 1:
            obs = np.expand_dims(obs, 0)
        if len(actions.shape) == 1:
            actions = np.expand_dims(actions, 0)
        elif len(actions.shape) == 0:
            # one discrete action
            actions = np.expand_dims(actions, 0)

        feed_dict = {self.generator_obs_ph: obs, self.generator_acs_ph: actions}
        reward = sess.run(self.reward_op, feed_dict)
        return reward

class TransitionClassifierMDPO(object):
    def __init__(self, sess, observation_space, action_space, hidden_size,
                 entcoeff=0.001, scope="adversary", normalize=True):
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
        self.expert_obs_ph = tf.placeholder(observation_space.dtype, (None,) + self.observation_shape,
                                            name="expert_observations_ph")
        self.expert_acs_ph = tf.placeholder(action_space.dtype, (None,) + self.actions_shape,
                                            name="expert_actions_ph")
        # Build graph
        generator_logits = self.build_graph(self.generator_obs_ph, self.generator_acs_ph, reuse=False)
        expert_logits = self.build_graph(self.expert_obs_ph, self.expert_acs_ph, reuse=True)
        # Build accuracy
        generator_acc = tf.reduce_mean(tf.cast(tf.nn.sigmoid(generator_logits) < 0.5, tf.float32))
        expert_acc = tf.reduce_mean(tf.cast(tf.nn.sigmoid(expert_logits) > 0.5, tf.float32))
        # Build regression loss
        # let x = logits, z = targets.
        # z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
        generator_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=generator_logits,
                                                                 labels=tf.zeros_like(generator_logits))
        generator_loss = tf.reduce_mean(generator_loss)
        expert_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=expert_logits,
                                                              labels=tf.ones_like(expert_logits))
        expert_loss = tf.reduce_mean(expert_loss)
        # Build entropy loss
        logits = tf.concat([generator_logits, expert_logits], 0)
        entropy = tf.reduce_mean(logit_bernoulli_entropy(logits))
        entropy_loss = -entcoeff * entropy
        # Loss + Accuracy terms
        self.losses = [generator_loss, expert_loss, entropy, entropy_loss, generator_acc, expert_acc]
        self.loss_name = ["generator_loss", "expert_loss", "entropy", "entropy_loss", "generator_acc", "expert_acc"]
        self.total_loss = generator_loss + expert_loss + entropy_loss
        # Build Reward for policy
        self.reward_op = -tf.log(1 - tf.nn.sigmoid(generator_logits) + 1e-8)
        var_list = self.get_trainable_variables()
        rewards_optimizer = tf.train.AdamOptimizer(learning_rate=3e-4)
        rewards_train_op = rewards_optimizer.minimize(self.total_loss, var_list=var_list)
        self.train = tf_util.function(
            [self.generator_obs_ph, self.generator_acs_ph, self.expert_obs_ph, self.expert_acs_ph],
            rewards_train_op)
        # self.lossandgrad = tf_util.function(
        #     [self.generator_obs_ph, self.generator_acs_ph, self.expert_obs_ph, self.expert_acs_ph],
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
            logits = tf.contrib.layers.fully_connected(p_h2, 1, activation_fn=tf.identity)
        return logits

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
