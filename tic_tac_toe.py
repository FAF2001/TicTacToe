import random
import numpy as np
from collections import defaultdict

# TicTacToe class
class TicTacToe:
    def __init__(self):
        self.board = [' ' for _ in range(9)]  # Empty 3x3 board
        self.current_winner = None  # Keep track of winner!
    
    def print_board(self):
        # Print the board in a user-friendly way
        for row in [self.board[i:i+3] for i in range(0, 9, 3)]:
            print("| " + " | ".join(row) + " |")
    
    @staticmethod
    def print_board_nums():
        # Print board with position numbers for reference
        number_board = [[str(i) for i in range(j, j+3)] for j in range(0, 9, 3)]
        for row in number_board:
            print("| " + " | ".join(row) + " |")
    
    def available_moves(self):
        return [i for i, spot in enumerate(self.board) if spot == ' ']
    
    def empty_squares(self):
        return ' ' in self.board
    
    def make_move(self, square, letter):
        if self.board[square] == ' ':
            self.board[square] = letter
            if self.winner(square, letter):
                self.current_winner = letter
            return True
        return False

    def winner(self, square, letter):
        # Check the row, column, and diagonal
        row_ind = square // 3
        row = self.board[row_ind*3:(row_ind+1)*3]
        if all([spot == letter for spot in row]):
            return True
        
        col_ind = square % 3
        column = [self.board[col_ind + i*3] for i in range(3)]
        if all([spot == letter for spot in column]):
            return True
        
        # Diagonal check
        if square % 2 == 0:
            diagonal1 = [self.board[i] for i in [0, 4, 8]]
            if all([spot == letter for spot in diagonal1]):
                return True
            
            diagonal2 = [self.board[i] for i in [2, 4, 6]]
            if all([spot == letter for spot in diagonal2]):
                return True
        
        return False

    def reset(self):
        # Reset the game state to play again
        self.board = [' ' for _ in range(9)]
        self.current_winner = None
    
    # Function to canonicalize the board state (handle all symmetrical versions)
    @staticmethod
    def canonical_form(board):
        boards = []
        b = np.array(board).reshape(3, 3)
        for k in range(4):
            rotated = np.rot90(b, k)
            boards.append(tuple(rotated.flatten()))
            boards.append(tuple(np.fliplr(rotated).flatten()))
        return min(boards)


# QLearningAgent class
class QLearningAgent:
    def __init__(self, epsilon=0.1, alpha=0.5, gamma=0.9):
        self.epsilon = epsilon  # Exploration rate
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.q_table = {}  # Q-table for storing state-action values
    
    def get_q(self, state, action):
        # If the state-action pair doesn't exist, initialize it
        if (state, action) not in self.q_table:
            self.q_table[(state, action)] = 0.0
        return self.q_table[(state, action)]
    
    def update_q(self, state, action, reward, next_state, done):
        best_q_next = max([self.get_q(next_state, a) for a in range(9)]) if not done else 0
        self.q_table[(state, action)] = self.get_q(state, action) + self.alpha * (reward + self.gamma * best_q_next - self.get_q(state, action))
    
    def choose_action(self, state, available_moves):
        if np.random.rand() < self.epsilon:
            return random.choice(available_moves)  # Explore: random move
        else:
            # Exploit: Choose the best action based on current Q-values
            q_values = [self.get_q(state, a) for a in available_moves]
            max_q = max(q_values)
            best_actions = [a for a, q in zip(available_moves, q_values) if q == max_q]
            return random.choice(best_actions)


# Training function
def train_agent(episodes=90000, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.9995, epsilon_min=0.1):
    game = TicTacToe()
    q_table = defaultdict(lambda: np.zeros(9))  # 9 actions (positions on the board)

    def get_state(board):
        return ''.join(board)

    for episode in range(episodes):
        game.reset()
        state = get_state(game.board)
        done = False

        while not done:
            if random.random() < epsilon:
                action = random.choice(game.available_moves())
            else:
                q_values = q_table[state]
                # Mask invalid actions
                for i in range(9):
                    if i not in game.available_moves():
                        q_values[i] = -np.inf
                action = np.argmax(q_values)

            valid_move = game.make_move(action, 'X')
            if not valid_move:
                reward = -1
                done = True
                continue

            if game.current_winner == 'X':
                reward = 1
                done = True
            elif not game.empty_squares():
                reward = 0.5  # draw
                done = True
            else:
                reward = 0

            next_state = get_state(game.board)
            best_next_action = np.max(q_table[next_state])

            # Q-learning update
            q_table[state][action] = q_table[state][action] + alpha * (reward + gamma * best_next_action - q_table[state][action])
            state = next_state

        # Decay epsilon
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        # Show progress
        if (episode + 1) % 1000 == 0:
            print(f"Episode {episode+1}/{episodes} — Epsilon: {epsilon:.3f}")

    print("✅ Training complete!")
    return q_table


# Playing function
def play_against_agent(q_table):
    game = TicTacToe()
    game.reset()
    
    def get_state(board):
        return ''.join(board)

    print("Let's play! You are 'O'. The agent is 'X'.")
    #game.print_board()

    while True:
        # Agent's move (X)
        state = get_state(game.board)
        q_values = q_table[state]
        # Mask invalid actions
        for i in range(9):
            if i not in game.available_moves():
                q_values[i] = -np.inf
        action = np.argmax(q_values)
        game.make_move(action, 'X')
        print("\nAgent played:")
        game.print_board()

        if game.current_winner == 'X':
            print("❌ Agent wins!")
            break
        if not game.empty_squares():
            print("🤝 It's a draw!")
            break

        # Your move (O)
        valid = False
        while not valid:
            try:
                move = int(input("Your move (0-8): "))
                valid = game.make_move(move, 'O')
                if not valid:
                    print("Invalid move. Try again.")
            except:
                print("Invalid input. Try again.")
        game.print_board()

        if game.current_winner == 'O':
            print("🎉 You win!")
            break
        if not game.empty_squares():
            print("🤝 It's a draw!")
            break


# Train agent and play against it
trained_agent = train_agent()

# Uncomment to play the game:
play_against_agent(trained_agent)
