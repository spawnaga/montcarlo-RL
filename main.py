import csv
import pickle
import random as rnd
import sqlite3
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
from collections import deque
from numpy.random import default_rng
import datetime
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from ib_insync import IB, util, ContFuture
import signal
import sys


class Market:
    """Class for handling market data"""

    def __init__(self, proxy=None, ibkr=False, database_path=r".\NQ_ticks.db"):
        self.df = None
        self.state_df = None
        self.db = sqlite3.connect(database_path)
        self.scaler = MinMaxScaler()
        self.offset = 0

        if ibkr:
            if proxy is None:
                self.ib = IB()

            else:
                self.ib = proxy

            self.checkIfReconnect()
            self.update_data(True)

    def checkIfReconnect(self):
        if not self.ib.isConnected() or not self.ib.client.isConnected():
            self.ib.disconnect()
            self.ib.connect('127.0.0.1', 7496, np.random.randint(0, 1000))

    def download_data(self, symbol='NQ', exchange='CME', length=1):
        self.contract = ContFuture(symbol, exchange)
        self.ib.qualifyContracts(self.contract)

        # Download historical data using reqHistoricalData
        self.bars = self.ib.reqHistoricalData(
            self.contract, endDateTime='', durationStr=f'{length} D',
            barSizeSetting='30 secs', whatToShow='TRADES',
            useRTH=False
        )
        # Create a DataFrame from the downloaded data
        df = util.df(self.bars)
        df = df.drop('date', axis=1)
        # df = self.get_analysis(df)
        df.reset_index(inplace=True, drop=True)
        return df

    def update_data(self, ibkr=False, chunk=None):
        if ibkr:
            df = self.download_data()
        else:
            df = chunk
        self.df = df
        self.state_df = df
        for col in self.state_df.columns:
            if not col == 'contract' and pd.api.types.is_numeric_dtype(self.state_df[col].dtype):
                self.state_df = self.state_df[
                    self.state_df[col].between(0, 300) | self.state_df[col].between(11000, 14000) | self.state_df[
                        col].between(-10, 10)].reset_index(drop=True)
        self.standardize_data()

    def standardize_data(self):

        def scale_column(col):
            if col.empty:
                raise ValueError("Input column is empty.")
            elif pd.api.types.is_datetime64_any_dtype(col.dtype):
                return col, None
            elif col.min() >= 0 and col.max() <= 300:
                scaler = MinMaxScaler(feature_range=(0, 1))
            elif col.min() >= 11000 and col.max() <= 14000:
                scaler = MinMaxScaler(feature_range=(0, 1))
            elif col.min() >= -10 and col.max() <= 10:
                scaler = MinMaxScaler(feature_range=(0, 1))
            else:
                return None, col.argmin()
            return scaler.fit_transform(col.values.reshape(-1, 1)), None

        for col in self.state_df.columns:
            if not col == 'contract':
                col_scaled, _ = scale_column(self.state_df[col])
                if col_scaled is not None:
                    self.state_df[col] = col_scaled
                else:
                    self.state_df = self.state_df.drop(_, axis=0)
                    self.state_df = self.state_df.reset_index(drop=True)
                    self.standardize_data()

    def get_state(self, i=-1, numContracts=0):
        """Method for getting current state of market"""
        global holdings
        if numContracts > 0:
            holdings = 1
        elif numContracts < 0:
            holdings = -1
        elif numContracts == 0:
            holdings = 0
        state = self.state_df.iloc[i, :].values
        holdings_array = np.array([holdings])
        state = np.concatenate([state, np.array(holdings_array)])
        state = state.reshape(1, -1)
        return state

    def load_data_chunk(self, file_path, table_name, chunksize):

        query = f'SELECT * FROM {table_name} LIMIT {chunksize} OFFSET {self.offset}'
        df = pd.read_sql_query(query, self.db)
        df.rename(columns={"lastPrice": 'close'}, inplace=True)
        df.dropna(inplace=True)
        self.update_data(False, df)
        self.offset += chunksize

    def get_df(self):
        """Method for returning the DataFrame of market data"""
        return self.df

    def disconnect(self):
        """Method for disconnect"""
        self.ib.disconnect()


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


class LSTMNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, output_size=3, batch_size=64):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, h0=None, c0=None):
        x = x.reshape(1, x.shape[0], x.shape[1])
        seq_len, batch_size, input_size = x.size()

        if h0 is None:
            h0 = torch.zeros(self.num_layers, seq_len, self.hidden_size).to(x.device)
        if c0 is None:
            c0 = torch.zeros(self.num_layers, seq_len, self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out)
        return out.squeeze(0)


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

    def choose_action(self, state, price, account_balance):
        max_num_contracts = int(min(account_balance / (price['close'] * self.contract_multiplier),
                                    np.iinfo(np.int64).max))

        if max_num_contracts <= 0:
            return 0, 0  # Return 'hold' action and 0 contracts

        if np.random.rand() <= self.epsilon:
            action = np.random.randint(0, 3)  # Hold, Buy (Long), or Sell (Short)
            num_contracts = self.rng.integers(1, max_num_contracts + 1,
                                              dtype='int64')  # Use the random number generator
            return action, num_contracts

        state = [value for value in state if not isinstance(value, pd.Timestamp)]
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

        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.LongTensor(np.array(actions)).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(1).to(device)

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
        # Save the model
        torch.save(self.q_net.state_dict(), file_path)

    def load_model(self, file_path):
        print(file_path)
        try:
            if os.path.isfile(file_path):
                # Check the file size of the model to be loaded
                file_size = os.path.getsize(file_path)
                print(f"Model to be loaded with file size: {file_size} bytes")

                self.q_net.load_state_dict(torch.load(file_path, map_location=device))
                self.target_net.load_state_dict(torch.load(file_path, map_location=device))
            else:
                print(f"Model file not found: {file_path}")
        except RuntimeError as e:
            print(f"Error loading model: {e}")


def main():
    def signal_handler(sig, frame):
        print("Interrupt received, saving the model before exiting.")
        agent.save_model(pre_trained_model_path)
        sys.exit(0)

    iteration = 100
    # Register the signal handler
    signal.signal(signal.SIGINT, signal_handler)

    # Instantiate the Market class
    market = Market(database_path=r'.\NQ_ticks.db')

    # Set the table_name and chunksize
    table_name = 'NQ_market_depth'
    chunksize = 10000

    while True:
        # Load a chunk of data using the load_data_chunk method
        market.load_data_chunk(table_name, table_name, chunksize)

        # Check if there are no more chunks left in the .db file
        if market.df.empty:
            break
        input_size = market.state_df.shape[1]
        output_size = 3  # Hold, Buy (Long), or Sell (Short)
        agent = TradingAgent(input_size, output_size)

        # Load a pre-trained model before starting the training loop, if it exists
        pre_trained_model_path = r".\model.pth"
        agent.load_model(pre_trained_model_path)

        # Load the saved replay buffer, if it exists
        replay_buffer_file_path = r".\replay_buffer.pkl"
        agent.load_replay_buffer(replay_buffer_file_path)

        starting_balance = 1000000
        episode = 0

        for ep in range(iteration):  # Train for iterations episodes per chunk

            if iteration > 25:
                iteration -= 5

            # Generate preprocessed data
            data = market.state_df
            df = market.df

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
            log_file = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".csv"
            log_file = os.path.join(r'.\trades_logs', log_file)
            with open(log_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(
                    ['Iteration', 'Close Price Before', 'Action', 'Position', 'Contracts', 'Reward', 'Total PNL',
                     'Balance After', 'Close Price After'])

            for t in range(len(data)):
                action, new_num_contracts = agent.choose_action(state, price, account_balance)
                next_state = data.iloc[t]
                next_price = df.iloc[t]
                reward = 0

                if action == 1:  # Buy (Long)
                    if position is None or position == 'short':
                        position = 'long'
                        entry_price_standard = next_price['close']
                        entry_price_nonstandard = next_price['close']
                        num_contracts = new_num_contracts
                elif action == 2:  # Sell (Short)
                    if position is None or position == 'long':
                        position = 'short'
                        entry_price_standard = next_price['close']
                        entry_price_nonstandard = next_price['close']
                        num_contracts = new_num_contracts

                if position == 'long':
                    trade_profit_standard = (next_price['close'] - entry_price_standard)
                    trade_profit_nonstandard = (next_price['close'] - entry_price_nonstandard)
                elif position == 'short':
                    trade_profit_standard = (entry_price_standard - next_price['close'])
                    trade_profit_nonstandard = (entry_price_nonstandard - next_price['close'])

                reward += (trade_profit_standard*num_contracts)
                account_balance += (trade_profit_nonstandard*20*num_contracts)

                done = (t == len(data) - 1)
                experience = (state.values, action, reward, next_state.values, done)
                agent.replay_buffer.add(experience)
                agent.train()
                state = next_state
                price = next_price
                total_reward += reward
                total_reward_nonstandard += trade_profit_nonstandard

                with open(log_file, mode='a') as f:
                    f.write(
                        f"{episode},{price['close']},{action},{position},"
                        f"{num_contracts},{reward},"
                        f"{(trade_profit_nonstandard*20*num_contracts)},"
                        f"{account_balance},{next_price['close']}\n")

                if done:
                    rewards = [experience[2] for experience in agent.replay_buffer.buffer]
                    std_dev = np.std(rewards)
                    print(
                        f"Episode: {episode}, action: {action}, Total Reward: {total_reward}, Final Balance: "
                        f"{account_balance}, Epsilon: {agent.epsilon}, Standard Deviation: {std_dev}")
                    agent.update_epsilon()

                    # Training for 100 episodes on the current chunk
                    if episode % 100 == 0:
                        agent.update_target_net()

                    # Logging
                    print(
                        f"{episode},{price['close']},{action},{position},"
                        f"{num_contracts},{reward},{(trade_profit_standard*num_contracts)},"
                        f"{(trade_profit_nonstandard*20*num_contracts)},"
                        f"{account_balance},{next_price['close']}\n")

                    episode += 1

            # Save the trained model after the last episode
            agent.save_model(pre_trained_model_path)
            agent.save_replay_buffer(replay_buffer_file_path)

    print("Training finished.")


if __name__ == "__main__":
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    main()
