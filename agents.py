import numpy as np
from tqdm import tqdm
import gymnasium as gym
import matplotlib.pyplot as plt
from collections import deque
import random
from tensorflow.keras.models import Sequential  # To compose multiple Layers
# Fully-Connected and Flattening layer
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.layers import Activation  # Activation functions
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import clone_model


class RLAgent:

    def __init__(self,
                 epsilon,
                 epochs):
        self.rewards = []
        self.rewards_rm = []
        self.successes = []
        self.successes_rm = []
        self.epsilon = epsilon
        self.epochs = epochs
        self.env = gym.make("LunarLander-v2")
        observation, info = self.env.reset(seed=42)

    def reduce_epsilon(self,
                       epoch,
                       method='linear',
                       burn_in=1,
                       epsilon_reduce=0.0001,
                       end_epsilon_reduction=10000,
                       reduction_factor=0.995):
        if method == 'linear':
            if burn_in <= epoch <= end_epsilon_reduction:
                self.epsilon -= epsilon_reduce
        elif method == 'geometric':
            self.epsilon *= reduction_factor  # Reduce epsilon

    ##### PLOTTING #####

    def plot_rewards(self, filepath, ylim=400):
        plt.plot(list(range(self.epochs)), self.rewards, label='Rewards')
        plt.plot(list(range(self.epochs)), self.rewards_rm,
                 label='Running mean of rewards (last 300 episodes)')
        plt.title('Rewards over time')
        plt.axhline(200, linestyle='dashed', color='black',
                    label='Threshold for solution')
        plt.ylim(0, ylim)
        plt.legend()
        plt.savefig(filepath)
        plt.show()

    def plot_successes(self, filepath):
        plt.plot(list(range(self.epochs)), self.successes_rm,
                 label='Running mean of successes (>=200 points)')
        plt.title('% of successes over time')
        plt.ylim(0, 1)
        plt.legend()
        plt.savefig(filepath)
        plt.show()


class RLAgentQTable(RLAgent):

    lunar_lander_limits = [(-1.5, 1.5), (-1.5, 1.5),
                           (-5, 5), (-5, 5), (-3.1415927, 3.1415927), (-5, 5)]

    def __init__(self,
                 epsilon=1,
                 num_bins=20,
                 epochs=20000,
                 alpha=0.8,
                 gamma=0.9):
        super().__init__(epsilon, epochs)
        self.alpha = alpha
        self.gamma = gamma
        q_table_shape = (num_bins, num_bins, num_bins, num_bins,
                         num_bins, num_bins, 2, 2, self.env.action_space.n)
        self.q_table = np.zeros(q_table_shape)
        self.bins = self.create_bins(num_bins)

    ##### DISCRETIZATION #####

    @classmethod
    def create_bins(cls, num_bins_per_action=10):
        bins_x_coord = np.linspace(
            cls.lunar_lander_limits[0][0], cls.lunar_lander_limits[0][1], num_bins_per_action)
        bins_y_coord = np.linspace(
            cls.lunar_lander_limits[1][0], cls.lunar_lander_limits[1][1], num_bins_per_action)
        bins_x_linear_vel = np.linspace(
            cls.lunar_lander_limits[2][0], cls.lunar_lander_limits[2][1], num_bins_per_action)
        bins_y_linear_vel = np.linspace(
            cls.lunar_lander_limits[3][0], cls.lunar_lander_limits[3][1], num_bins_per_action)
        bins_angle = np.linspace(
            cls.lunar_lander_limits[4][0], cls.lunar_lander_limits[4][1], num_bins_per_action)
        bins_angular_vel = np.linspace(
            cls.lunar_lander_limits[5][0], cls.lunar_lander_limits[5][1], num_bins_per_action)
        bins = np.array([bins_x_coord,
                        bins_y_coord,
                        bins_x_linear_vel,
                        bins_y_linear_vel,
                        bins_angle,
                        bins_angular_vel])
        return bins

    @classmethod
    def trim_outliers(cls, observations):
        trimmed_observations = []
        for i in range(len(observations)):
            el = observations[i]
            if el < cls.lunar_lander_limits[i][0]:
                el = cls.lunar_lander_limits[i][0] + 1e-3
            elif el > cls.lunar_lander_limits[i][1]:
                el = cls.lunar_lander_limits[i][1] - 1e-3
            trimmed_observations.append(el)
        return trimmed_observations

    @classmethod
    def discretize_observation(cls, observations, bins):
        binned_observations = []
        trimmed_observations = cls.trim_outliers(observations)
        for i, observation in enumerate(trimmed_observations):
            discretized_observation = np.digitize(observation, bins[i])
            binned_observations.append(discretized_observation)
        return binned_observations

    ##### LEARNING #####

    def epsilon_greedy_action_selection(self, discrete_state):
        random_number = np.random.random()
        # EXPLOITATION
        if random_number > self.epsilon:
            action = np.argmax(self.q_table[discrete_state])
        # EXPLORATION
        else:
            # Return a random 0,1,2,3 action
            action = np.random.randint(0, self.env.action_space.n)
        return action

    def compute_next_q_value(self, old_q_value, reward, next_optimal_q_value):
        return old_q_value + self.alpha * (reward + self.gamma * next_optimal_q_value - old_q_value)

    def learn(self):
        for epoch in tqdm(range(self.epochs)):
            initial_state, info = self.env.reset()  # get the initial observation
            discretized_state = tuple(self.discretize_observation(initial_state[0:6], self.bins) + [
                                      # map the observation to the bins
                                      int(initial_state[6]), int(initial_state[7])])
            terminated = False
            truncated = False
            points = 0  # store result

            while not terminated and not truncated:
                action = self.epsilon_greedy_action_selection(
                    discretized_state)  # epsilon-greedy action selection
                observation, reward, terminated, truncated, info = self.env.step(
                    action)  # perform action and get next state

                next_state_discretized = tuple(RLAgentQTable.discretize_observation(
                    # map the next observation to the bins
                    observation[0:6], self.bins) + [int(observation[6]), int(observation[7])])
                # get the old Q-Value from the Q-Table
                old_q_value = self.q_table[discretized_state + (action,)]
                # Get the next optimal Q-Value
                next_optimal_q_value = np.max(
                    self.q_table[next_state_discretized])

                next_q = self.compute_next_q_value(
                    old_q_value, reward, next_optimal_q_value)  # Compute next Q-Value
                # Insert next Q-Value into the table
                self.q_table[discretized_state + (action,)] = next_q

                discretized_state = next_state_discretized  # Update the old state
                points += 1

                if points >= 400:
                    terminated = True  # No need to keep the game running too long

            self.reduce_epsilon(epoch)  # Reduce epsilon
            # log overall achieved points for the current epoch
            self.rewards.append(points)
            # Compute running mean points over the last 30 epochs
            self.rewards_rm.append(np.mean(self.rewards[-300:]))
            # Has the agent scored at least 200 points?
            self.successes.append(points >= 200)
            self.successes_rm.append(np.mean(self.successes[-300:]))

        self.env.close()


class RLAgentDQN(RLAgent):

    def __init__(self,
                 epsilon=1,
                 epochs=1000,
                 alpha=0.001,
                 gamma=0.95):
        super().__init__(epsilon, epochs)
        self.gamma = gamma
        self.model = Sequential()
        self.model.add(Input(shape=(self.env.observation_space.shape[0],)))
        self.model.add(Dense(16))
        self.model.add(Activation('relu'))
        self.model.add(Dense(32))
        self.model.add(Activation('relu'))
        self.model.add(Dense(self.env.action_space.n))
        self.model.add(Activation('linear'))
        self.target_model = clone_model(self.model)
        self.replay_buffer = deque(maxlen=20000)
        self.model.compile(loss='mse', optimizer=Adam(learning_rate=alpha))

    #### LEARNING ####

    def epsilon_greedy_action_selection(self, observation):
        if np.random.random() > self.epsilon:
            observation = observation.reshape([1, 8])
            # perform the prediction on the observation
            prediction = self.model.predict(observation, verbose=0)
            # Chose the action with the higher value
            action = np.argmax(prediction)
        else:
            # Else use random action
            action = np.random.randint(0, self.env.action_space.n)
        return action

    def replay(self, batch_size):
        # As long as the buffer has not enough elements we do nothing
        if len(self.replay_buffer) < batch_size:
            return
        # Take a random sample from the buffer with size batch_size
        samples = random.sample(self.replay_buffer, batch_size)
        # to store the targets predicted by the target network for training
        target_batch = []
        # Efficient way to handle the sample by using the zip functionality
        zipped_samples = list(zip(*samples))
        states, actions, rewards, new_states, terminateds, truncateds = zipped_samples
        # Predict targets for all states from the sample
        targets = self.target_model.predict(np.array(states), verbose=0)
        # Predict Q-Values for all new states from the sample
        q_values = self.model.predict(np.array(new_states), verbose=0)
        # Now we loop over all predicted values to compute the actual targets
        for i in range(batch_size):
            # Take the maximum Q-Value for each sample
            q_value = max(q_values[i])
            # Store the ith target in order to update it according to the formula
            target = targets[i].copy()
            if terminateds[i] or truncateds[i]:
                target[actions[i]] = rewards[i]
            else:
                target[actions[i]] = rewards[i] + q_value * self.gamma
            target_batch.append(target)
        # Fit the model based on the states and the updated targets for 1 epoch
        self.model.fit(np.array(states), np.array(
            target_batch), epochs=1, verbose=0)

    def update_model_handler(self, epoch, update_target_model=10):
        if epoch > 0 and epoch % update_target_model == 0:
            self.target_model.set_weights(self.model.get_weights())

    def learn(self):
        best_so_far = 0
        for epoch in tqdm(range(self.epochs)):
            observation, info = self.env.reset()  # Get inital state
            terminated = False
            truncated = False
            points = 0
            while not terminated and not truncated:  # as long current run is active
                # Select action acc. to strategy
                action = self.epsilon_greedy_action_selection(observation)
                # Perform action and get next state
                next_observation, reward, terminated, truncated, info = self.env.step(
                    action)
                # Update the replay buffer
                self.replay_buffer.append(
                    (observation, action, reward, next_observation, terminated, truncated))
                observation = next_observation  # update the observation
                points += 1
                # Most important step! Training the model by replaying
                self.replay(32)
            self.reduce_epsilon(epoch, method='geometric')  # Reduce epsilon
            # log overall achieved points for the current epoch
            self.rewards.append(points)
            # Compute running mean points over the last 30 epochs
            self.rewards_rm.append(np.mean(self.rewards[-300:]))
            # Has the agent scored at least 200 points?
            self.successes.append(points >= 200)
            self.successes_rm.append(np.mean(self.successes[-300:]))
            # Check if we need to update the target model
            self.update_model_handler(epoch)
            if points > best_so_far:
                best_so_far = points
            if epoch % 25 == 0:
                print(f'''{epoch}: Points reached: {
                      points} - epsilon: {self.epsilon} - Best: {best_so_far}''')
        # Saving model weights
        self.model.save_weights('out/dqn_model_weights.weights.h5')
