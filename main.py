import pickle
import random as rnd
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
from collections import deque
from numpy.random import default_rng
import datetime


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return rnd.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class QNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=64):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def preprocess_data(file_path):
    nonstandard_df = pd.read_csv(file_path)

    # Preprocessing steps, such as handling missing data or creating additional features, should be done here
    nonstandard_df = nonstandard_df.dropna()

    # Standardize the data
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(nonstandard_df)
    standardized_df = pd.DataFrame(standardized_data, columns=nonstandard_df.columns)

    return standardized_df, nonstandard_df


class TradingAgent:
    def __init__(self, input_size, output_size, hidden_size=64, lr=0.001, gamma=0.99, buffer_capacity=100000,
                 batch_size=64):
        self.q_net = QNetwork(input_size, output_size, hidden_size).to(device)
        self.target_net = QNetwork(input_size, output_size, hidden_size).to(device)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.rng = default_rng()
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        self.batch_size = batch_size
        self.contract_multiplier = 20
        self.reward_scaling = 1e-6

    def choose_action(self, state, account_balance):
        max_num_contracts = int(min(account_balance / (state['close'] * self.contract_multiplier),
                                    np.iinfo(np.int64).max))

        if max_num_contracts <= 0:
            return 0, 0  # Return 'hold' action and 0 contracts

        if np.random.rand() <= self.epsilon:
            action = np.random.randint(0, 3)  # Hold, Buy (Long), or Sell (Short)
            num_contracts = self.rng.integers(1, max_num_contracts + 1,
                                              dtype='int64')  # Use the random number generator
            return action, num_contracts

        q_values = self.q_net(torch.FloatTensor(state).to(device))
        action = torch.argmax(q_values).item()
        num_contracts = self.rng.integers(1, max_num_contracts + 1, dtype='int64')  # Use the random number generator
        return action, num_contracts

    def update_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def update_target_net(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(device)  # Change here
        actions = torch.LongTensor(np.array(actions)).unsqueeze(1).to(device)  # Change here
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(device)  # Change here
        next_states = torch.FloatTensor(np.array(next_states)).to(device)  # Change here
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(1).to(device)  # Change here

        q_values = self.q_net(states).gather(1, actions)
        next_q_values = self.target_net(next_states).detach().max(1)[0].unsqueeze(1)
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = self.criterion(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_replay_buffer(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self.replay_buffer, f)

    def load_replay_buffer(self, file_path):
        if os.path.isfile(file_path):
            with open(file_path, 'rb') as f:
                self.replay_buffer = pickle.load(f)
        else:
            print(f"Replay buffer file not found: {file_path}")

    def save_model(self, file_path):
        torch.save(self.q_net.state_dict(), file_path)

    def load_model(self, file_path):
        if os.path.isfile(file_path):
            self.q_net.load_state_dict(torch.load(file_path))
            self.target_net.load_state_dict(torch.load(file_path))
        else:
            print(f"Model file not found: {file_path}")


def main():
    file_path = "NQ.csv"
    data, df = preprocess_data(file_path)
    input_size = data.shape[1]
    output_size = 3  # Hold, Buy (Long), or Sell (Short)
    agent = TradingAgent(input_size, output_size)

    # Load a pre-trained model before starting the training loop, if it exists
    pre_trained_model_path = "model.pth"
    agent.load_model(pre_trained_model_path)

    # Load the saved replay buffer, if it exists
    replay_buffer_file_path = "replay_buffer.pkl"
    agent.load_replay_buffer(replay_buffer_file_path)

    num_episodes = 100
    starting_balance = 1000000
    log_file = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".csv"
    with open(log_file, mode='w') as f:
        f.write(
            "Iteration, Balance Before, Close Price Before, Action, Contracts, Position, Reward, Balance After, "
            "Close Price After\n")
    for episode in range(num_episodes):
        state = data.iloc[0]
        price = df.iloc[0]
        total_reward = 0
        total_reward_nonstandard = 0
        account_balance = starting_balance
        position = None
        entry_price_standard = 0
        entry_price_nonstandard = 0
        num_contracts = 0
        trade_profit_standard = 0
        trade_profit_nonstandard = 0
        for t in range(1, len(data)):
            action, new_num_contracts = agent.choose_action(state, account_balance)
            next_state = data.iloc[t]
            next_price = df.iloc[t]
            reward = 0

            if action == 1:  # Buy (Long)
                if position is None or position == 'short':
                    position = 'long'
                    entry_price_standard = next_state['close']
                    entry_price_nonstandard = next_price['close']
                    num_contracts = new_num_contracts
            elif action == 2:  # Sell (Short)
                if position is None or position == 'long':
                    position = 'short'
                    entry_price_standard = next_state['close']
                    entry_price_nonstandard = next_price['close']
                    num_contracts = new_num_contracts

            if position == 'long':
                trade_profit_standard = (next_state['close'] - entry_price_standard)
                trade_profit_nonstandard = (next_price['close'] - entry_price_nonstandard)
            elif position == 'short':
                trade_profit_standard = (entry_price_standard - next_state['close'])
                trade_profit_nonstandard = (entry_price_nonstandard - next_price['close'])

            reward += trade_profit_standard
            account_balance += trade_profit_nonstandard

            done = (t == len(data) - 1)
            experience = (state.values, action, reward, next_state.values, done)
            agent.replay_buffer.add(experience)
            agent.train()
            state = next_state
            price = next_price
            total_reward += reward
            total_reward_nonstandard += account_balance

            if done:
                print(
                    f"Episode: {episode}, action: {action}, Total Reward: {total_reward_nonstandard}, Final Balance: "
                    f"{account_balance}, Epsilon: {agent.epsilon},")
                agent.update_epsilon()
                if episode % 10 == 0:
                    agent.update_target_net()

                # Logging
                with open(log_file, mode='a') as f:
                    f.write(
                        f"{episode},{account_balance - trade_profit_standard},{price['close']},{action},{num_contracts},"
                        f"{position},{reward},{account_balance},{next_price['close']}\n")

        # Save the trained model after the last episode
        agent.save_model(pre_trained_model_path)
        agent.save_replay_buffer(pre_trained_model_path)


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    main()
