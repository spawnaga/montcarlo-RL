# Monte Carlo Reinforcement Learning for Trading

This is a simple implementation of a trading agent that uses a Monte Carlo Reinforcement Learning algorithm to learn to trade a futures contract. The agent uses OpenAI Gym's trading environment to simulate trading and PyTorch to build a neural network that is used as a function approximator for the Q-function.

## Requirements

- Python 3.6 or later
- PyTorch 1.5.0 or later
- pandas 1.0.5 or later
- scikit-learn 0.23.1 or later
- ib_insync 0.9.72 or later
- Jupyter Notebook (optional)

## Usage

To run the code, simply execute the `main()` function in `monte_carlo_rl.py`. The script will download historical market data, use a csv or a SQL file for a futures contract and use it to train the trading agent. The training process can take several hours depending on the length of the historical data and the complexity of the neural network.

## Disclaimer

This code is provided for educational purposes only and should not be used for actual trading. Trading futures contracts is risky and can result in large financial losses.