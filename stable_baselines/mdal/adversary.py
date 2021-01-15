"""
Reference: https://github.com/openai/imitation
I follow the architecture from the official repository
"""
import gym
import tensorflow as tf
import numpy as np

from stable_baselines.common.mpi_running_mean_std import RunningMeanStd as MpiRunningMeanStd

from stable_baselines.common.running_mean_std import RunningMeanStd, RunningMinMax
from stable_baselines.common import tf_util as tf_util
from stable_baselines.common import zipsame



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
        self.policy_obs_ph = tf.placeholder(observation_space.dtype, (None,) + self.observation_shape,
                                               name="observations_ph")
        self.policy_acs_ph = tf.placeholder(action_space.dtype, (None,) + self.actions_shape,
                                               name="actions_ph")
        self.policy_gammas_ph = tf.placeholder(tf.float32, (None, 1), name="gammas_ph")
        self.expert_obs_ph = tf.placeholder(observation_space.dtype, (None,) + self.observation_shape,
                                            name="expert_observations_ph")
        self.expert_acs_ph = tf.placeholder(action_space.dtype, (None,) + self.actions_shape,
                                            name="expert_actions_ph")
        self.expert_gammas_ph = tf.placeholder(tf.float32, (None, 1), name="gammas_ph")
        self.mix_obs_ph = tf.placeholder(observation_space.dtype, (None,) + self.observation_shape,
                                            name="expert_observations_ph")
        self.mix_acs_ph = tf.placeholder(action_space.dtype, (None,) + self.actions_shape,
                                            name="expert_actions_ph")


        # Build graph
        policy_rewards = self.build_graph(self.policy_obs_ph, self.policy_acs_ph, reuse=False)
        expert_rewards = self.build_graph(self.expert_obs_ph, self.expert_acs_ph, reuse=True)
        # generator_rewards = tf.math.sigmoid(generator_logits)
        # expert_rewards = tf.math.sigmoid(expert_logits)
        # policy_scaled_rewards = tf.multiply(policy_rewards, self.policy_gammas_ph)
        policy_scaled_rewards = policy_rewards
        # policy_value = (1-0.99) * tf.reduce_sum(policy_scaled_rewards)
        policy_value = tf.reduce_mean(policy_scaled_rewards)
        # expert_scaled_rewards = tf.multiply(expert_rewards, self.expert_gammas_ph)
        expert_scaled_rewards = expert_rewards
        # expert_value = (1-0.99) * tf.reduce_sum(expert_scaled_rewards)
        expert_value = tf.reduce_mean(expert_scaled_rewards)

        # alpha = tf.random.uniform([], 0.0, 1.0, observation_space.dtype)
        # generator_obs_mix = tf.reduce_mean(self.generator_obs_ph, axis=0, keepdims=True)
        # generator_acs_mix = tf.reduce_mean(self.generator_acs_ph, axis=0, keepdims=True)
        # expert_obs_mix = tf.reduce_mean(self.expert_obs_ph, axis=0, keepdims=True)
        # expert_acs_mix = tf.reduce_mean(self.expert_acs_ph, axis=0, keepdims=True)
        # generator_obs_mix = (1-0.99) * tf.reduce_sum(tf.cast(self.generator_gammas_ph, observation_space.dtype) * self.generator_obs_ph, axis=0, keepdims=True)
        # generator_acs_mix = (1-0.99) * tf.reduce_sum(tf.cast(self.generator_gammas_ph, action_space.dtype) * self.generator_acs_ph, axis=0, keepdims=True)
        # expert_obs_mix = (1-0.99) * tf.reduce_sum(tf.cast(self.expert_gammas_ph, observation_space.dtype) * self.expert_obs_ph, axis=0, keepdims=True)
        # expert_acs_mix = (1-0.99) * tf.reduce_sum(tf.cast(self.expert_gammas_ph, action_space.dtype) * self.expert_acs_ph, axis=0, keepdims=True)
        # mixture_obs = alpha * generator_obs_mix + (1 - alpha) * tf.reduce_mean(expert_obs_mix)
        # mixture_acs = tf.cast(alpha, action_space.dtype) * generator_acs_mix\
        #               + tf.cast((1 - alpha), action_space.dtype) * expert_acs_mix
        mixture_rewards = self.build_graph(self.mix_obs_ph, self.mix_acs_ph, reuse=True)
        grads = tf.gradients(mixture_rewards, [self.mix_obs_ph, self.mix_acs_ph])[0]
        norm = tf.cast(tf.sqrt(tf.reduce_sum(tf.square(grads), axis=1)), tf.float32)
        grad_reg = tf.reduce_mean(tf.square(norm - 1.0))
        grad_reg_coef = 1.0
        grad_reg_loss = grad_reg_coef * grad_reg

        rewards = tf.concat([policy_rewards, expert_rewards], 0)

        rewards_reg = - tf.reduce_mean(logit_bernoulli_entropy(rewards))
        rewards_reg_coef = 0.001

        # rewards_reg = tf.reduce_sum(tf.square(rewards))
        # rewards_reg_coef = 0.01

        rewards_reg_loss = rewards_reg_coef * rewards_reg
        policy_loss = policy_value - expert_value
        loss = policy_loss + grad_reg_loss + rewards_reg_loss

        # Loss + Accuracy terms
        self.losses = [loss]
        self.loss_name = ["generator_loss", "expert_loss", "entropy", "entropy_loss", "generator_acc", "expert_acc"]
        # self.total_loss = loss
        # Build Reward for policy
        self.reward_op = tf.stop_gradient(policy_rewards)
        # self.reward_op = generator_rewards


        var_list = self.get_trainable_variables()
        rewards_optimizer = tf.train.AdamOptimizer(learning_rate=3e-4)
        # rewards_optimizer = tf.train.AdamOptimizer(learning_rate=1e-5)

        # rewards_optimizer = tf.train.AdamOptimizer(learning_rate=3e-4, beta1=0)
        # rewards_optimizer = tf.train.AdamOptimizer(learning_rate=1e-3, beta1=0.5)

        # rewards_optimizer = tf.train.AdamOptimizer(learning_rate=1e-2)

        # grads, vars = zip(*rewards_optimizer.compute_gradients(loss, var_list=var_list))
        # accum_vars = [tf.Variable(tf.zeros_like(var.initialized_value()), trainable=False) for var in var_list]
        # accumulation_counter = tf.Variable(0.0, trainable=False)

        # zero_ops = [var.assign(tf.zeros_like(var)) for var in accum_vars]
        # zero_ops.append(accumulation_counter.assign(0.0))

        # gvs = rewards_optimizer.compute_gradients(loss, var_list)
        # accumulate_ops = [accum_vars[i].assign_add(gv[0]) for i, gv in enumerate(gvs)]
        # accumulate_ops.append(accumulation_counter.assign_add(1.0))

        # train_step = rewards_optimizer.apply_gradients([(accum_vars[i] / accumulation_counter, gv[1]) for i, gv in enumerate(gvs)])

        # grads, vars = list(zip(*grads_and_vars))
        # grads, norm = tf.clip_by_global_norm(grads, 300.0)
        # rewards_train_op = rewards_optimizer.apply_gradients(zip(grads, vars))
        # norm = tf.constant(0.)
        rewards_train_op = rewards_optimizer.minimize(loss, var_list=var_list)

        # rewards_train_op = [rewards_train_op, norm]

        # self.zero_grad = tf_util.function([], zero_ops)
        # self.compute_grads = tf_util.function(
        #     [self.generator_obs_ph, self.generator_acs_ph, self.generator_gammas_ph,
        #      self.expert_obs_ph, self.expert_acs_ph, self.expert_gammas_ph], accumulate_ops)
        # self.train = tf_util.function([], train_step)
        # print_op = tf.print("Value diff:", policy_value - expert_value, "Grad Regularizer:", grad_reg)
        print_op = tf.no_op()
        self.train = tf_util.function(
            [self.policy_obs_ph, self.policy_acs_ph, self.policy_gammas_ph,
             self.expert_obs_ph, self.expert_acs_ph, self.expert_gammas_ph,
             self.mix_obs_ph, self.mix_acs_ph], [rewards_train_op, print_op])


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
                obs = tf.cast(obs_ph, tf.float32)

            if self.discrete_actions:
                one_hot_actions = tf.one_hot(acs_ph, self.n_actions)
                actions_ph = tf.cast(one_hot_actions, tf.float32)
            else:
                actions_ph = acs_ph

            _input = tf.concat([obs, actions_ph], axis=1)  # concatenate the two input -> form a transition
            p_h1 = tf.contrib.layers.fully_connected(_input, self.hidden_size, activation_fn=tf.nn.tanh)
            p_h2 = tf.contrib.layers.fully_connected(p_h1, self.hidden_size, activation_fn=tf.nn.tanh)
            # rewards = tf.contrib.layers.fully_connected(p_h2, 1, activation_fn=tf.nn.tanh)
            # rewards = tf.contrib.layers.fully_connected(p_h2, 1, activation_fn=tf.math.sigmoid)
            rewards = tf.contrib.layers.fully_connected(p_h2, 1, activation_fn=tf.identity)
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

        feed_dict = {self.policy_obs_ph: obs, self.policy_acs_ph: actions}
        reward = self.sess.run(self.reward_op, feed_dict)
        return reward

class NeuralAdversaryTRPO(object):
    def __init__(self, sess, observation_space, action_space, hidden_size=64, entcoeff=0.001,
                 scope="adversary", normalize=True):
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
        self.policy_obs_ph = tf.placeholder(observation_space.dtype, (None,) + self.observation_shape,
                                               name="observations_ph")
        self.policy_acs_ph = tf.placeholder(action_space.dtype, (None,) + self.actions_shape,
                                               name="actions_ph")
        self.policy_gammas_ph = tf.placeholder(tf.float32, (None, 1), name="gammas_ph")
        self.expert_obs_ph = tf.placeholder(observation_space.dtype, (None,) + self.observation_shape,
                                            name="expert_observations_ph")
        self.expert_acs_ph = tf.placeholder(action_space.dtype, (None,) + self.actions_shape,
                                            name="expert_actions_ph")
        self.expert_gammas_ph = tf.placeholder(tf.float32, (None, 1), name="gammas_ph")
        self.mix_obs_ph = tf.placeholder(observation_space.dtype, (None,) + self.observation_shape,
                                            name="expert_observations_ph")
        self.mix_acs_ph = tf.placeholder(action_space.dtype, (None,) + self.actions_shape,
                                            name="expert_actions_ph")


        # Build graph
        policy_rewards = self.build_graph(self.policy_obs_ph, self.policy_acs_ph, reuse=False)
        expert_rewards = self.build_graph(self.expert_obs_ph, self.expert_acs_ph, reuse=True)
        # policy_rewards = tf.math.sigmoid(policy_logits)
        # expert_rewards = tf.math.sigmoid(expert_logits)
        # policy_scaled_rewards = tf.multiply(policy_rewards, self.policy_gammas_ph)
        policy_scaled_rewards = policy_rewards
        # policy_value = tf.reduce_sum(policy_scaled_rewards)
        policy_value = tf.reduce_mean(policy_scaled_rewards)
        # expert_scaled_rewards = tf.multiply(expert_rewards, self.expert_gammas_ph)
        expert_scaled_rewards = expert_rewards
        # expert_value = tf.reduce_sum(expert_scaled_rewards)
        expert_value = tf.reduce_mean(expert_scaled_rewards)

        # alpha = tf.random.uniform([], 0.0, 1.0, observation_space.dtype)
        # generator_obs_mix = tf.reduce_mean(self.generator_obs_ph, axis=0, keepdims=True)
        # generator_acs_mix = tf.reduce_mean(self.generator_acs_ph, axis=0, keepdims=True)
        # expert_obs_mix = tf.reduce_mean(self.expert_obs_ph, axis=0, keepdims=True)
        # expert_acs_mix = tf.reduce_mean(self.expert_acs_ph, axis=0, keepdims=True)
        # generator_obs_mix = (1-0.99) * tf.reduce_sum(tf.cast(self.generator_gammas_ph, observation_space.dtype) * self.generator_obs_ph, axis=0, keepdims=True)
        # generator_acs_mix = (1-0.99) * tf.reduce_sum(tf.cast(self.generator_gammas_ph, action_space.dtype) * self.generator_acs_ph, axis=0, keepdims=True)
        # expert_obs_mix = (1-0.99) * tf.reduce_sum(tf.cast(self.expert_gammas_ph, observation_space.dtype) * self.expert_obs_ph, axis=0, keepdims=True)
        # expert_acs_mix = (1-0.99) * tf.reduce_sum(tf.cast(self.expert_gammas_ph, action_space.dtype) * self.expert_acs_ph, axis=0, keepdims=True)
        # mixture_obs = alpha * generator_obs_mix + (1 - alpha) * tf.reduce_mean(expert_obs_mix)
        # mixture_acs = tf.cast(alpha, action_space.dtype) * generator_acs_mix\
        #               + tf.cast((1 - alpha), action_space.dtype) * expert_acs_mix
        mixture_rewards = self.build_graph(self.mix_obs_ph, self.mix_acs_ph, reuse=True)
        grads = tf.gradients(mixture_rewards, [self.mix_obs_ph, self.mix_acs_ph])[0]
        norm = tf.cast(tf.sqrt(tf.reduce_sum(tf.square(grads), axis=1)), tf.float32)
        grad_reg = tf.reduce_mean(tf.square(norm - 1.0))
        grad_reg_coef = 0.0
        grad_reg_loss = grad_reg_coef * grad_reg

        rewards = tf.concat([policy_rewards, expert_rewards], 0)

        rewards_reg = - tf.reduce_mean(logit_bernoulli_entropy(rewards))
        # rewards_reg = tf.reduce_sum(tf.square(rewards))

        rewards_reg_coef = 0.0

        rewards_reg_loss = rewards_reg_coef * rewards_reg

        policy_loss = policy_value - expert_value
        self.total_loss = policy_loss + grad_reg_loss + rewards_reg_loss

        # Loss + Accuracy terms
        self.losses = []
        self.loss_name = ["generator_loss", "expert_loss", "entropy", "entropy_loss", "generator_acc", "expert_acc"]
        # self.total_loss = loss
        # Build Reward for policy
        # self.reward_op = tf.stop_gradient(policy_rewards)
        self.reward_op = tf.stop_gradient(tf.clip_by_value(policy_rewards, -1.0, 1.0))
        # self.reward_op = generator_rewards

        print_op = tf.print("Policy loss:", policy_loss,
                            "GradReg", grad_reg,
                            "rewards_abs_mean", tf.reduce_mean(tf.abs(rewards)), "rewards_std", tf.math.reduce_std(rewards),
                            "abs_max", tf.math.reduce_max(tf.abs(rewards)))
        var_list = self.get_trainable_variables()
        self.lossandgrad = tf_util.function(
            [self.policy_obs_ph, self.policy_acs_ph, self.policy_gammas_ph,
             self.expert_obs_ph, self.expert_acs_ph, self.expert_gammas_ph,
             self.mix_obs_ph, self.mix_acs_ph],
            self.losses + [print_op] + [tf_util.flatgrad(self.total_loss, var_list)])

        # print_op = tf.print("Value diff:", policy_value - expert_value, "Grad Regularizer:", grad_reg)
        # print_op = tf.no_op()
        # self.train = tf_util.function(
        #     [self.policy_obs_ph, self.policy_acs_ph, self.policy_gammas_ph,
        #      self.expert_obs_ph, self.expert_acs_ph, self.expert_gammas_ph,
        #      self.mix_obs_ph, self.mix_acs_ph], [rewards_train_op, print_op])


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
                    self.obs_rms = MpiRunningMeanStd(shape=self.observation_shape)
                obs = (tf.cast(obs_ph, tf.float32) - self.obs_rms.mean) / tf.cast(self.obs_rms.std, tf.float32)
            else:
                obs = tf.cast(obs_ph, tf.float32)

            if self.discrete_actions:
                one_hot_actions = tf.one_hot(acs_ph, self.n_actions)
                actions_ph = tf.cast(one_hot_actions, tf.float32)
            else:
                actions_ph = acs_ph

            _input = tf.concat([obs, actions_ph], axis=1)  # concatenate the two input -> form a transition
            p_h1 = tf.contrib.layers.fully_connected(_input, self.hidden_size, activation_fn=tf.nn.tanh)
            p_h2 = tf.contrib.layers.fully_connected(p_h1, self.hidden_size, activation_fn=tf.nn.tanh)
            # rewards = tf.contrib.layers.fully_connected(p_h2, 1, activation_fn=tf.nn.tanh)
            # rewards = tf.contrib.layers.fully_connected(p_h2, 1, activation_fn=tf.math.sigmoid)
            rewards = tf.contrib.layers.fully_connected(p_h2, 1, activation_fn=tf.identity)
            # last_layer_init = tf.contrib.layers.variance_scaling_initializer(factor=0.1, mode='FAN_AVG', uniform=True)
            # rewards = tf.contrib.layers.fully_connected(p_h2, 1, activation_fn=tf.identity, weights_initializer=last_layer_init)

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

        feed_dict = {self.policy_obs_ph: obs, self.policy_acs_ph: actions}
        reward = self.sess.run(self.reward_op, feed_dict)
        return reward


class NeuralAdversaryMDPO(object):
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
        self.policy_obs_ph = tf.placeholder(observation_space.dtype, (None,) + self.observation_shape,
                                               name="observations_ph")
        self.policy_acs_ph = tf.placeholder(action_space.dtype, (None,) + self.actions_shape,
                                               name="actions_ph")
        self.policy_gammas_ph = tf.placeholder(tf.float32, (None, 1), name="gammas_ph")
        self.expert_obs_ph = tf.placeholder(observation_space.dtype, (None,) + self.observation_shape,
                                            name="expert_observations_ph")
        self.expert_acs_ph = tf.placeholder(action_space.dtype, (None,) + self.actions_shape,
                                            name="expert_actions_ph")
        self.expert_gammas_ph = tf.placeholder(tf.float32, (None, 1), name="gammas_ph")
        self.mix_obs_ph = tf.placeholder(observation_space.dtype, (None,) + self.observation_shape,
                                            name="expert_observations_ph")
        self.mix_acs_ph = tf.placeholder(action_space.dtype, (None,) + self.actions_shape,
                                            name="expert_actions_ph")


        # Build graph
        policy_rewards = self.build_graph(self.policy_obs_ph, self.policy_acs_ph, reuse=False, scope=self.scope)
        expert_rewards = self.build_graph(self.expert_obs_ph, self.expert_acs_ph, reuse=True, scope=self.scope)
        old_policy_rewards = self.build_graph(self.policy_obs_ph, self.policy_acs_ph, reuse=False, scope="oldreward")
        old_expert_rewards = self.build_graph(self.expert_obs_ph, self.expert_acs_ph, reuse=True, scope="oldreward")

        # generator_rewards = tf.math.sigmoid(generator_logits)
        # expert_rewards = tf.math.sigmoid(expert_logits)
        # policy_scaled_rewards = tf.multiply(policy_rewards, self.policy_gammas_ph)
        policy_scaled_rewards = policy_rewards
        # policy_value = (1-0.99) * tf.reduce_sum(policy_scaled_rewards)
        policy_value = tf.reduce_mean(policy_scaled_rewards)
        # expert_scaled_rewards = tf.multiply(expert_rewards, self.expert_gammas_ph)
        expert_scaled_rewards = expert_rewards
        # expert_value = (1-0.99) * tf.reduce_sum(expert_scaled_rewards)
        expert_value = tf.reduce_mean(expert_scaled_rewards)

        # alpha = tf.random.uniform([], 0.0, 1.0, observation_space.dtype)
        # generator_obs_mix = tf.reduce_mean(self.generator_obs_ph, axis=0, keepdims=True)
        # generator_acs_mix = tf.reduce_mean(self.generator_acs_ph, axis=0, keepdims=True)
        # expert_obs_mix = tf.reduce_mean(self.expert_obs_ph, axis=0, keepdims=True)
        # expert_acs_mix = tf.reduce_mean(self.expert_acs_ph, axis=0, keepdims=True)
        # generator_obs_mix = (1-0.99) * tf.reduce_sum(tf.cast(self.generator_gammas_ph, observation_space.dtype) * self.generator_obs_ph, axis=0, keepdims=True)
        # generator_acs_mix = (1-0.99) * tf.reduce_sum(tf.cast(self.generator_gammas_ph, action_space.dtype) * self.generator_acs_ph, axis=0, keepdims=True)
        # expert_obs_mix = (1-0.99) * tf.reduce_sum(tf.cast(self.expert_gammas_ph, observation_space.dtype) * self.expert_obs_ph, axis=0, keepdims=True)
        # expert_acs_mix = (1-0.99) * tf.reduce_sum(tf.cast(self.expert_gammas_ph, action_space.dtype) * self.expert_acs_ph, axis=0, keepdims=True)
        # mixture_obs = alpha * generator_obs_mix + (1 - alpha) * tf.reduce_mean(expert_obs_mix)
        # mixture_acs = tf.cast(alpha, action_space.dtype) * generator_acs_mix\
        #               + tf.cast((1 - alpha), action_space.dtype) * expert_acs_mix
        mixture_rewards = self.build_graph(self.mix_obs_ph, self.mix_acs_ph, reuse=True, scope=self.scope)
        grads = tf.gradients(mixture_rewards, [self.mix_obs_ph, self.mix_acs_ph])[0]
        norm = tf.cast(tf.sqrt(tf.reduce_sum(tf.square(grads), axis=1)), tf.float32)
        grad_reg = tf.reduce_mean(tf.square(norm - 1.0))
        grad_reg_coef = 10
        grad_reg_loss = grad_reg_coef * grad_reg
        #
        # rewards = tf.concat([policy_rewards, expert_rewards], 0)
        #
        # rewards_reg = - tf.reduce_mean(logit_bernoulli_entropy(rewards))
        # rewards_reg_coef = 0.001
        #


        rewards = tf.concat([policy_rewards, expert_rewards], 0)

        policy_clipped_rewards = tf.clip_by_value(policy_rewards, -10.0, 10.0)
        expert_clipped_rewards = tf.clip_by_value(expert_rewards, -10.0, 10.0)
        clipped_rewards = tf.concat([policy_clipped_rewards, expert_clipped_rewards], 0)

        # rewards_reg_loss = rewards_reg_coef * rewards_reg
        old_rewards = tf.concat([old_policy_rewards, old_expert_rewards], 0)

        old_policy_clipped_rewards = tf.clip_by_value(old_policy_rewards, -10.0, 10.0)
        old_expert_clipped_rewards = tf.clip_by_value(old_expert_rewards, -10.0, 10.0)
        old_clipped_rewards = tf.concat([old_policy_clipped_rewards, old_expert_clipped_rewards], 0)




        # rewards_reg_coef = 0.01
        # bregman = tf.reduce_mean(tf_util.huber_loss(old_rewards - rewards))
        bregman = tf.reduce_mean(tf.square(tf.stop_gradient(old_clipped_rewards) - rewards))

        bregman_coeff = 100
        bregman_loss = bregman_coeff * bregman
        #
        # stepsize = 0.001

        rewards_reg = - tf.reduce_mean(logit_bernoulli_entropy(rewards))
        rewards_reg_coeff = 0.00
        # rewards_reg = tf.reduce_mean(tf.square(rewards))
        # rewards_reg_coeff = 1
        rewards_reg_loss = rewards_reg_coeff * rewards_reg

        # rewards_reg_loss = rewards_reg_coef * rewards_reg

        # policy_loss = policy_value - expert_value
        # rewards_gradient = (tf.gradients(tf.reduce_mean(old_policy_clipped_rewards), [old_policy_rewards])[0]
        #                - tf.gradients(tf.reduce_mean(old_expert_clipped_rewards), [old_expert_rewards])[0])
        old_policy_loss = tf.reduce_mean(old_policy_rewards) - tf.reduce_mean(old_expert_rewards)
        old_rewards_gradient = tf.concat(tf.gradients(old_policy_loss, [old_policy_rewards, old_expert_rewards]), axis=0)

        policy_loss = tf.reduce_sum(tf.multiply(tf.stop_gradient(old_rewards_gradient), clipped_rewards))

        loss = policy_loss + bregman_loss + rewards_reg_loss + grad_reg_loss

        # Loss + Accuracy terms
        self.losses = [loss]
        self.loss_name = ["generator_loss", "expert_loss", "entropy", "entropy_loss", "generator_acc", "expert_acc"]
        # self.total_loss = loss
        # Build Reward for policy
        self.reward_op = old_policy_clipped_rewards
        # self.reward_op = tf.stop_gradient(tf.clip_by_value(policy_rewards, -1.0, 1.0))
        # self.reward_op = generator_rewards

        self.update_old_rewards = \
            tf_util.function([], [], updates=[tf.assign(oldv, newv) for (oldv, newv) in
                                              zipsame(tf_util.get_globals_vars("oldreward"),
                                                      tf_util.get_globals_vars(self.scope))])

        var_list = self.get_trainable_variables()
        # rewards_optimizer = tf.train.AdamOptimizer(learning_rate=1e-5, epsilon=1e-5)
        # rewards_optimizer = tf.train.AdamOptimizer(learning_rate=3e-4, beta1=0.9895193, beta2=0.9999, epsilon=1e-5)

        # rewards_optimizer = tf.train.AdamOptimizer(learning_rate=3e-4, beta1=0, beta2=0.9)
        rewards_optimizer = tf.train.AdamOptimizer(learning_rate=3e-4)

        # rewards_optimizer = tf.train.AdamOptimizer(learning_rate=1e-3, beta1=0.5)

        # rewards_optimizer = tf.train.AdamOptimizer(learning_rate=1e-2)

        # accum_vars = [tf.Variable(tf.zeros_like(var.initialized_value()), trainable=False) for var in var_list]
        # accumulation_counter = tf.Variable(0.0, trainable=False)

        # zero_ops = [var.assign(tf.zeros_like(var)) for var in accum_vars]
        # zero_ops.append(accumulation_counter.assign(0.0))

        # gvs = rewards_optimizer.compute_gradients(loss, var_list)
        # accumulate_ops = [accum_vars[i].assign_add(gv[0]) for i, gv in enumerate(gvs)]
        # accumulate_ops.append(accumulation_counter.assign_add(1.0))

        # train_step = rewards_optimizer.apply_gradients([(accum_vars[i] / accumulation_counter, gv[1]) for i, gv in enumerate(gvs)])
        grads, vars = zip(*rewards_optimizer.compute_gradients(loss, var_list=var_list))
        grads, norm = tf.clip_by_global_norm(grads, 1e7)
        # grads, norm = tf.clip_by_global_norm(grads, 5.0)

        rewards_train_op = rewards_optimizer.apply_gradients(zip(grads, vars))
        # norm = tf.constant(0.)
        # rewards_train_op = rewards_optimizer.minimize(loss, var_list=var_list)

        # rewards_train_op = [rewards_train_op, norm]

        # self.zero_grad = tf_util.function([], zero_ops)
        # self.compute_grads = tf_util.function(
        #     [self.generator_obs_ph, self.generator_acs_ph, self.generator_gammas_ph,
        #      self.expert_obs_ph, self.expert_acs_ph, self.expert_gammas_ph], accumulate_ops)
        # self.train = tf_util.function([], train_step)
        linear_approximation = tf.stop_gradient(tf.reduce_sum(tf.multiply(old_rewards_gradient, rewards - old_rewards)))
        # print_op = tf.print("Value diff:", linear_approximation, "Bregman:", bregman_loss,
        #                     "MD objective",  linear_approximation + bregman_loss,
        #                     "Weight GradNorm:", norm, "Loss norm", tf.norm(old_rewards_gradient),
        #                     "rewards_mean", tf.reduce_mean(rewards), "rewards_std", tf.math.reduce_std(rewards),
        #                     "rewards_abs_max", tf.math.reduce_max(rewards))
        # print_op = tf.print("old_rewards_gradient", tf.shape(old_rewards_gradient), "rewards",tf.shape(rewards))
        print_op = tf.no_op()




        # var_list = self.get_trainable_variables()

        # grads, vars = zip(*rewards_optimizer.compute_gradients(loss, var_list=var_list))
        # grads, norm = tf.clip_by_global_norm(grads, 300.0)
        # rewards_train_op = rewards_optimizer.apply_gradients(zip(grads, vars))


        self.train = tf_util.function(
            [self.policy_obs_ph, self.policy_acs_ph, self.policy_gammas_ph,
             self.expert_obs_ph, self.expert_acs_ph, self.expert_gammas_ph,
             self.mix_obs_ph, self.mix_acs_ph], [rewards_train_op, print_op])


    def build_graph(self, obs_ph, acs_ph, reuse=False, scope=None):
        """
        build the graph

        :param obs_ph: (tf.Tensor) the observation placeholder
        :param acs_ph: (tf.Tensor) the action placeholder
        :param reuse: (bool)
        :return: (tf.Tensor) the graph output
        """
        with tf.variable_scope(scope):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            if self.normalize:
                with tf.variable_scope("obfilter"):
                    self.obs_rms = RunningMeanStd(shape=self.observation_shape)
                obs = (tf.cast(obs_ph, tf.float32) - self.obs_rms.mean) / tf.cast(tf.sqrt(self.obs_rms.var), tf.float32)
            else:
                obs = tf.cast(obs_ph, tf.float32)

            if self.discrete_actions:
                one_hot_actions = tf.one_hot(acs_ph, self.n_actions)
                actions_ph = tf.cast(one_hot_actions, tf.float32)
            else:
                actions_ph = acs_ph

                _input = tf.concat([obs, actions_ph], axis=1)  # concatenate the two input -> form a transition
                p_h1 = tf.contrib.layers.fully_connected(_input, self.hidden_size, activation_fn=tf.nn.tanh)
                p_h2 = tf.contrib.layers.fully_connected(p_h1, self.hidden_size, activation_fn=tf.nn.tanh)
                rewards = tf.contrib.layers.fully_connected(p_h2, 1, activation_fn=tf.identity)


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

        feed_dict = {self.policy_obs_ph: obs, self.policy_acs_ph: actions}
        reward = self.sess.run(self.reward_op, feed_dict)
        return reward

class NeuralAdversaryMD(object):
    def __init__(self, sess, observation_space, action_space, hidden_size=64, entcoeff=0.001,
                 scope="adversary", normalize=True):
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
        self.policy_obs_ph = tf.placeholder(observation_space.dtype, (None,) + self.observation_shape,
                                               name="observations_ph")
        self.policy_acs_ph = tf.placeholder(action_space.dtype, (None,) + self.actions_shape,
                                               name="actions_ph")
        self.policy_gammas_ph = tf.placeholder(tf.float32, (None, 1), name="gammas_ph")
        self.expert_obs_ph = tf.placeholder(observation_space.dtype, (None,) + self.observation_shape,
                                            name="expert_observations_ph")
        self.expert_acs_ph = tf.placeholder(action_space.dtype, (None,) + self.actions_shape,
                                            name="expert_actions_ph")
        self.expert_gammas_ph = tf.placeholder(tf.float32, (None, 1), name="expert_gammas_ph")
        self.mix_obs_ph = tf.placeholder(observation_space.dtype, (None,) + self.observation_shape,
                                            name="expert_observations_ph")
        self.mix_acs_ph = tf.placeholder(action_space.dtype, (None,) + self.actions_shape,
                                            name="expert_actions_ph")

        if self.normalize:
            with tf.variable_scope("obfilter"):
                self.obs_rms = MpiRunningMeanStd(shape=self.observation_shape)

        # Build graph
        policy_rewards = self.build_graph(self.policy_obs_ph, self.policy_acs_ph, reuse=False, scope=self.scope)
        expert_rewards = self.build_graph(self.expert_obs_ph, self.expert_acs_ph, reuse=True, scope=self.scope)
        old_policy_rewards = self.build_graph(self.policy_obs_ph, self.policy_acs_ph, reuse=False, scope="oldreward")
        old_expert_rewards = self.build_graph(self.expert_obs_ph, self.expert_acs_ph, reuse=True, scope="oldreward")


        # policy_scaled_rewards = tf.multiply(policy_rewards, self.policy_gammas_ph)
        policy_scaled_rewards = policy_rewards
        # policy_value = tf.reduce_sum(policy_scaled_rewards)
        policy_value = tf.reduce_mean(policy_scaled_rewards)
        # expert_scaled_rewards = tf.multiply(expert_rewards, self.expert_gammas_ph)
        expert_scaled_rewards = expert_rewards
        # expert_value = tf.reduce_sum(expert_scaled_rewards)
        expert_value = tf.reduce_mean(expert_scaled_rewards)


        mixture_rewards = self.build_graph(self.mix_obs_ph, self.mix_acs_ph, reuse=True, scope=self.scope)
        grads = tf.gradients(mixture_rewards, [self.mix_obs_ph, self.mix_acs_ph])[0]
        norm = tf.cast(tf.sqrt(tf.reduce_sum(tf.square(grads), axis=1)), tf.float32)
        lipschitz_reg = tf.reduce_mean(tf.square(norm - 1.0))
        lipschitz_coef = 1.0
        lipschitz_loss = lipschitz_coef * lipschitz_reg
        #

        rewards = tf.concat([policy_rewards, expert_rewards], 0)

        policy_clipped_rewards = tf.clip_by_value(policy_rewards, -10.0, 10.0)
        expert_clipped_rewards = tf.clip_by_value(expert_rewards, -10.0, 10.0)
        clipped_rewards = tf.concat([policy_clipped_rewards, expert_clipped_rewards], 0)

        #
        # rewards_reg = - tf.reduce_mean(logit_bernoulli_entropy(rewards))
        # rewards_reg_coef = 0.001
        #
        # rewards_reg = tf.reduce_sum(tf.square(rewards))

        # rewards_reg_coef = 0.01

        old_rewards = tf.concat([old_policy_rewards, old_expert_rewards], 0)

        old_policy_clipped_rewards = tf.clip_by_value(old_policy_rewards, -10.0, 10.0)
        old_expert_clipped_rewards = tf.clip_by_value(old_expert_rewards, -10.0, 10.0)
        old_clipped_rewards = tf.concat([old_policy_clipped_rewards, old_expert_clipped_rewards], 0)

        # bregman = tf.reduce_mean(tf_util.huber_loss(old_clipped_rewards - rewards))
        bregman = tf.reduce_mean(tf.square(tf.stop_gradient(old_clipped_rewards) - rewards))

        bregman_coeff = 100
        bregman_loss = bregman_coeff * bregman

        #
        # stepsize = 0.001


        # old_policy_loss = tf.reduce_mean(tf.multiply(old_policy_rewards, self.policy_gammas_ph))\
        #                   - tf.reduce_mean(tf.multiply(old_expert_rewards, self.expert_gammas_ph))

        new_policy_loss = tf.reduce_mean(tf.multiply(policy_rewards, self.policy_gammas_ph))\
                          - tf.reduce_mean(tf.multiply(expert_rewards, self.expert_gammas_ph))

        old_policy_loss = tf.reduce_mean(old_policy_rewards) - tf.reduce_mean(old_expert_rewards)
        old_rewards_gradient = tf.concat(tf.gradients(old_policy_loss, [old_policy_rewards, old_expert_rewards]), axis=0)

        policy_loss = tf.reduce_sum(tf.multiply(tf.stop_gradient(old_rewards_gradient), clipped_rewards))

        rewards_reg = - tf.reduce_mean(logit_bernoulli_entropy(rewards))
        rewards_reg_coeff = 0.001
        # rewards_reg = tf.reduce_mean(tf.square(rewards))
        # rewards_reg = tf.reduce_mean(tf_util.huber_loss(rewards))
        # rewards_reg_coeff = 0
        rewards_reg_loss = rewards_reg_coeff * rewards_reg

        # rewards_reg_loss = rewards_reg_coef * rewards_reg
        # policy_loss = policy_value - exp  ert_value
        self.total_loss = policy_loss + bregman_loss + rewards_reg_loss + lipschitz_loss

        # Loss + Accuracy terms
        self.losses = []
        self.loss_name = ["generator_loss", "expert_loss", "entropy", "entropy_loss", "generator_acc", "expert_acc"]
        # Build Reward for policy
        self.reward_op = tf.stop_gradient(old_policy_clipped_rewards)
        # self.reward_op = old_policy_rewards

        # self.reward_op = tf.clip_by_value(policy_rewards, -1.0, 1.0)
        # self.reward_op = generator_rewards




        self.update_old_rewards = \
            tf_util.function([], [], updates=[tf.assign(oldv, newv) for (oldv, newv) in
                                              zipsame(tf_util.get_globals_vars("oldreward"),
                                                      tf_util.get_globals_vars(self.scope))])

        var_list = self.get_trainable_variables()

        # clip_weights = [tf.assign(var, tf.clip_by_value(var,  -0.5, 0.5)) for var in var_list]
        clip_weights = tf.no_op()
        self.clip_weights = tf_util.function([], [clip_weights])

        # rewards_optimizer = tf.train.AdamOptimizer(learning_rate=3e-5)
        # grads, vars = zip(*rewards_optimizer.compute_gradients(self.total_loss, var_list=var_list))
        # rewards_train_op = rewards_optimizer.apply_gradients(zip(grads, vars))

        linear_approximation = tf.stop_gradient(tf.reduce_sum(tf.multiply(old_rewards_gradient, rewards - old_rewards)))


        grads = tf.gradients(self.total_loss, var_list)
        grads, norm = tf.clip_by_global_norm(grads, 1e7)


        # self.train = tf_util.function(
        #     [self.policy_obs_ph, self.policy_acs_ph, self.policy_gammas_ph,
        #      self.expert_obs_ph, self.expert_acs_ph, self.expert_gammas_ph,
        #      self.mix_obs_ph, self.mix_acs_ph], [rewards_train_op, print_op])

        loss_op = tf.concat(axis=0, values=[tf.reshape(grad if grad is not None else tf.zeros_like(v), [tf_util.numel(v)])
                                            for (v, grad) in zip(var_list, grads)])

        print_op = tf.print("Policy Loss:", new_policy_loss, "Bregman:", bregman_loss,
                            "MD objective",  linear_approximation + bregman_loss,
                            "GradNorm", norm,
                            "Loss norm", tf.norm(old_rewards_gradient),
                            "averageEnt", rewards_reg_loss,
                            "mean", tf.reduce_mean(tf.abs(rewards)), "std", tf.math.reduce_std(rewards),
                            "max", tf.math.reduce_max(tf.abs(rewards)))

        self.lossandgrad = tf_util.function(
            [self.policy_obs_ph, self.policy_acs_ph, self.policy_gammas_ph,
             self.expert_obs_ph, self.expert_acs_ph, self.expert_gammas_ph,
             self.mix_obs_ph, self.mix_acs_ph],
             # self.losses + [tf_util.flatgrad(self.total_loss, var_list)]) #, clip_norm=0.5, clip_by_global_norm=True)])
             self.losses + [print_op] + [loss_op])




        # print_op = tf.print("Value diff:", policy_value - expert_value, "Grad Regularizer:", grad_reg)
        # print_op = tf.no_op()
        # self.train = tf_util.function(
        #     [self.policy_obs_ph, self.policy_acs_ph, self.policy_gammas_ph,
        #      self.expert_obs_ph, self.expert_acs_ph, self.expert_gammas_ph,
        #      s, [rewards_train_op, print_op])


    def build_graph(self, obs_ph, acs_ph, reuse=False, scope=None):
        """
        build the graph

        :param obs_ph: (tf.Tensor) the observation placeholder
        :param acs_ph: (tf.Tensor) the action placeholder
        :param reuse: (bool)
        :return: (tf.Tensor) the graph output
        """
        with tf.variable_scope(scope):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            if self.normalize:
                obs = (tf.cast(obs_ph, tf.float32) - self.obs_rms.mean) / tf.cast(self.obs_rms.std, tf.float32)
            else:
                obs = tf.cast(obs_ph, tf.float32)

            if self.discrete_actions:
                one_hot_actions = tf.one_hot(acs_ph, self.n_actions)
                actions_ph = tf.cast(one_hot_actions, tf.float32)
            else:
                actions_ph = acs_ph

                _input = tf.concat([obs, actions_ph], axis=1)  # concatenate the two input -> form a transition
                p_h1 = tf.contrib.layers.fully_connected(_input, self.hidden_size, activation_fn=tf.nn.tanh)
                p_h2 = tf.contrib.layers.fully_connected(p_h1, self.hidden_size, activation_fn=tf.nn.tanh)
                # rewards = tf.contrib.layers.fully_connected(p_h2, 1, activation_fn=tf.nn.tanh)
                # rewards = tf.contrib.layers.fully_connected(p_h2, 1, activation_fn=tf.math.sigmoid)
                rewards = tf.contrib.layers.fully_connected(p_h2, 1, activation_fn=tf.identity)
                # last_layer_init = tf.contrib.layers.variance_scaling_initializer(factor=0.01, mode='FAN_AVG', uniform=True)
                # rewards = tf.contrib.layers.fully_connected(p_h2, 1, activation_fn=tf.identity, weights_initializer=last_layer_init)

        return rewards

    def get_trainable_variables(self):
        """
        Get all the trainable variables from the graph

        :return: ([tf.Tensor]) the variables
        """
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)# + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "oldreward")

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

        feed_dict = {self.policy_obs_ph: obs, self.policy_acs_ph: actions}
        reward = self.sess.run(self.reward_op, feed_dict)
        return reward
