# Tic-Tac-Toe Q-Learning Agent

This project implements a Q-learning agent that learns to play Tic-Tac-Toe against a human player.

## Overview

The project includes:
- A `TicTacToe` class that represents the game state and logic.
- A `QLearningAgent` class that learns and makes decisions based on the game state.
- A function `train_agent` to train the Q-learning agent over multiple episodes.
- A function `play_against_agent` to play the trained agent against a human.

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/FAF2001/TicTacToe.git
   ```

2. Install necessary dependencies (like `numpy`):
   ```bash
   pip install numpy
   ```

3. Train the agent:
   ```python
   trained_agent = train_agent(episodes=90000)
   ```

4. Play against the agent:
   ```python
   play_against_agent(trained_agent)
   ```

## How It Works

The agent is trained using Q-learning, a reinforcement learning technique. The agent learns by making moves and receiving feedback based on the game outcome (win, lose, draw). Over time, the agent refines its strategy to maximize its chances of winning.
