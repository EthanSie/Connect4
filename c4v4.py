import pygame
import sys
import numpy as np
import random
import time
import os
from collections import deque

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim

# Import tqdm for progress bar
from tqdm import tqdm

# ---------------------------
# Global Constants & Colors
# ---------------------------
ROW_COUNT = 6
COLUMN_COUNT = 7
SQUARESIZE = 100
RADIUS = int(SQUARESIZE / 2 - 5)
width = COLUMN_COUNT * SQUARESIZE
height = (ROW_COUNT + 1) * SQUARESIZE  # extra row for drop preview

# Colors (RGB)
BLUE   = (0, 0, 255)
BLACK  = (0, 0, 0)
RED    = (255, 0, 0)
YELLOW = (255, 255, 0)
WHITE  = (255, 255, 255)

# Model persistence filename for the DQN agent
MODEL_FILE = "dqn_connect4_model.pth"

# ---------------------------
# Connect 4 Environment
# ---------------------------
class Connect4Env:
    def __init__(self):
        self.rows = ROW_COUNT
        self.cols = COLUMN_COUNT
        self.board = np.zeros((self.rows, self.cols), dtype=int)

    def reset(self):
        self.board = np.zeros((self.rows, self.cols), dtype=int)
        return self.board.copy()

    def get_available_actions(self, board):
        """Return a list of columns (actions) that are not full."""
        return [col for col in range(self.cols) if board[0, col] == 0]

    def step(self, action, player):
        """
        Drop a piece in the given column for the given player.
        Returns (new_board, reward, done, info).
        
        - Drops the piece into the lowest available row.
        - Returns a reward of +1 for a win,
          0 for a draw,
          and a small penalty (-0.01) for each non–terminal move.
        - Illegal moves (should not occur in proper play) yield -10 and end the game.
        """
        if action not in self.get_available_actions(self.board):
            return self.board.copy(), -10, True, {"error": "Invalid move"}

        # Drop the piece into the lowest available row.
        for row in range(self.rows - 1, -1, -1):
            if self.board[row, action] == 0:
                self.board[row, action] = player
                break

        # Check for a win.
        if self.check_win(self.board, player):
            return self.board.copy(), 1, True, {"winner": player}

        # Check for a draw.
        if len(self.get_available_actions(self.board)) == 0:
            return self.board.copy(), 0, True, {"winner": 0}

        # Otherwise, apply a small penalty.
        return self.board.copy(), -0.01, False, {}

    def check_win(self, board, player):
        """Return True if the player has four in a row (horizontally, vertically, or diagonally)."""
        # Horizontal check.
        for r in range(self.rows):
            for c in range(self.cols - 3):
                if board[r, c] == player and board[r, c+1] == player and \
                   board[r, c+2] == player and board[r, c+3] == player:
                    return True
        # Vertical check.
        for c in range(self.cols):
            for r in range(self.rows - 3):
                if board[r, c] == player and board[r+1, c] == player and \
                   board[r+2, c] == player and board[r+3, c] == player:
                    return True
        # Positive diagonal.
        for r in range(self.rows - 3):
            for c in range(self.cols - 3):
                if board[r, c] == player and board[r+1, c+1] == player and \
                   board[r+2, c+2] == player and board[r+3, c+3] == player:
                    return True
        # Negative diagonal.
        for r in range(3, self.rows):
            for c in range(self.cols - 3):
                if board[r, c] == player and board[r-1, c+1] == player and \
                   board[r-2, c+2] == player and board[r-3, c+3] == player:
                    return True
        return False

# ---------------------------
# Canonical State Function
# ---------------------------
def canonical_state(board, player):
    """
    Return a canonical representation of the board from the perspective of the current player.
    If player==1, returns the board unchanged; if player==-1, returns board * -1.
    (The returned value is a tuple of tuples so it can be used as a key if needed.)
    """
    if player == 1:
        canon = board.copy()
    else:
        canon = board * -1
    return tuple(tuple(int(x) for x in row) for row in canon)

# ---------------------------
# Deep Q–Network Architecture
# ---------------------------
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        """
        A simple fully connected network:
          - input_size: number of inputs (here, 6*7 = 42)
          - output_size: number of actions (7 columns)
        """
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# ---------------------------
# DQN Agent with Experience Replay & Target Network
# ---------------------------
class DQNAgent:
    def __init__(self, input_size, output_size, lr=0.001, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.99995,
                 memory_size=10000, batch_size=64, target_update=1000):
        self.input_size = input_size
        self.output_size = output_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.steps_done = 0

        self.memory = deque(maxlen=memory_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = DQN(input_size, output_size).to(self.device)
        self.target_net = DQN(input_size, output_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

    def select_action(self, state, available_actions):
        """
        Given the current state (a numpy array of shape (6,7)) and available actions,
        select an action using epsilon–greedy exploration.
        """
        self.steps_done += 1
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        if random.random() < self.epsilon:
            return random.choice(available_actions)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state.flatten()).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor).cpu().numpy()[0]
            # Only consider Q–values for available actions.
            available_q = {a: q_values[a] for a in available_actions}
            return max(available_q, key=available_q.get)

    def store_transition(self, state, action, reward, next_state, done, next_avail_actions):
        self.memory.append((state, action, reward, next_state, done, next_avail_actions))

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones, next_avail_actions = zip(*transitions)
    
        # Convert list of arrays to a single numpy array before making a tensor
        states_np = np.array([s.flatten() for s in states])
        states = torch.FloatTensor(states_np).to(self.device)
    
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
    
        # Similarly convert next_states:
        next_states_np = np.array([s.flatten() for s in next_states])
        next_states = torch.FloatTensor(next_states_np).to(self.device)

        # Current Q-values for the actions taken.
        q_values = self.policy_net(states).gather(1, actions)

        # Compute target Q-values.
        with torch.no_grad():
            next_q_values = self.target_net(next_states)
            next_q_values_np = next_q_values.cpu().numpy()
            target_q_list = []
            for i in range(self.batch_size):
                avail = next_avail_actions[i]
                if len(avail) > 0:
                    max_q = max([next_q_values_np[i][a] for a in avail])
                else:
                    max_q = 0.0
                target_q_list.append(max_q)
            target_q = torch.FloatTensor(target_q_list).unsqueeze(1).to(self.device)
            expected_q = rewards + (1 - dones) * self.gamma * target_q

        loss = nn.MSELoss()(q_values, expected_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network periodically.
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

# ---------------------------
# Model Persistence Functions for DQN
# ---------------------------
def save_model_dqn(agent, filename=MODEL_FILE):
    torch.save(agent.policy_net.state_dict(), filename)
    print(f"Model saved to {filename}")

def load_model_dqn(agent, filename=MODEL_FILE):
    if os.path.exists(filename):
        agent.policy_net.load_state_dict(torch.load(filename, map_location=agent.device))
        agent.target_net.load_state_dict(agent.policy_net.state_dict())
        print(f"Loaded model from {filename}")
    else:
        print("No saved model found; starting fresh.")

# ---------------------------
# DQN Training Function (Self–play)
# ---------------------------
def train_dqn_agent(episodes=100000, agent=None):
    """
    Train the DQN agent using self–play.
    The agent plays both sides (using the canonical state transformation) so that
    learning is from the perspective of the moving player.
    A progress bar is shown during training.
    """
    env = Connect4Env()
    if agent is None:
        agent = DQNAgent(input_size=ROW_COUNT*COLUMN_COUNT, output_size=COLUMN_COUNT)
    print(f"Training DQN agent for {episodes} episodes...")
    for episode in tqdm(range(episodes), desc="Training DQN Agent"):
        board = env.reset()
        current_player = 1  # Start with player 1.
        done = False
        while not done:
            # Get state from current player's perspective.
            state = np.array(canonical_state(board, current_player), dtype=np.float32)
            available_actions = env.get_available_actions(board)
            action = agent.select_action(state, available_actions)
            new_board, reward, done, info = env.step(action, current_player)
            # Prepare next state (from the opponent's perspective) if not done.
            if not done:
                next_state = np.array(canonical_state(new_board, -current_player), dtype=np.float32)
                next_avail = env.get_available_actions(new_board)
            else:
                next_state = np.zeros((ROW_COUNT, COLUMN_COUNT), dtype=np.float32)
                next_avail = []
            # Store the transition.
            agent.store_transition(state, action, reward, next_state, done, next_avail)
            agent.optimize_model()
            board = new_board
            current_player *= -1  # Switch players.
        if (episode + 1) % 5000 == 0:
            print(f"Episode {episode+1}/{episodes} completed.")
    print("Training complete!")
    return agent

# ---------------------------
# Pygame Drawing Function
# ---------------------------
def draw_board_pygame(board, screen):
    """
    Draw the Connect 4 board.
    (The board is drawn so that row 0 is at the top and row ROW_COUNT-1 is at the bottom.)
    """
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT):
            pygame.draw.rect(screen, BLUE,
                             (c * SQUARESIZE, r * SQUARESIZE + SQUARESIZE, SQUARESIZE, SQUARESIZE))
            pygame.draw.circle(screen, BLACK,
                               (int(c * SQUARESIZE + SQUARESIZE / 2),
                                int(r * SQUARESIZE + SQUARESIZE + SQUARESIZE / 2)),
                               RADIUS)
    # Draw the pieces.
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT):
            if board[r][c] == 1:
                pygame.draw.circle(screen, RED,
                                   (int(c * SQUARESIZE + SQUARESIZE / 2),
                                    int(r * SQUARESIZE + SQUARESIZE + SQUARESIZE / 2)),
                                   RADIUS)
            elif board[r][c] == -1:
                pygame.draw.circle(screen, YELLOW,
                                   (int(c * SQUARESIZE + SQUARESIZE / 2),
                                    int(r * SQUARESIZE + SQUARESIZE + SQUARESIZE / 2)),
                                   RADIUS)
    pygame.display.update()

# ---------------------------
# Main Game Loop (Pygame)
# ---------------------------
def main():
    pygame.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Connect 4 – DQN Version")
    font = pygame.font.SysFont("monospace", 75)

    print("Choose game mode:")
    print("1: Human vs Human")
    print("2: Human vs DQN Agent (with additional training)")
    print("3: DQN Agent vs DQN Agent (with additional training)")
    print("4: Continuous Improvement (Self–play training)")
    print("5: Load and Play (without additional training)")
    mode = input("Enter mode (1, 2, 3, 4, or 5): ").strip()
    while mode not in ["1", "2", "3", "4", "5"]:
        mode = input("Enter mode (1, 2, 3, 4, or 5): ").strip()
    mode = int(mode)

    env = Connect4Env()
    board = env.reset()
    game_over = False
    current_player = 1  # 1 is Red, -1 is Yellow.
    agent = None

    # For modes that involve an agent, create a DQN agent and load its model.
    if mode in [2, 3, 4, 5]:
        agent = DQNAgent(input_size=ROW_COUNT*COLUMN_COUNT, output_size=COLUMN_COUNT)
        load_model_dqn(agent, MODEL_FILE)
        if mode == 2:
            # Human vs Agent: run a moderate training session before play.
            agent = train_dqn_agent(episodes=10000, agent=agent)
            save_model_dqn(agent, MODEL_FILE)
            print("Training complete. You are playing as Red (1). The Agent is Yellow (-1).")
        elif mode == 3:
            # Agent vs Agent: moderate training before play.
            agent = train_dqn_agent(episodes=10000, agent=agent)
            save_model_dqn(agent, MODEL_FILE)
            print("Training complete. Agent vs Agent mode.")
        elif mode == 4:
            # Continuous Improvement: long self–play training.
            agent = train_dqn_agent(episodes=100000, agent=agent)
            save_model_dqn(agent, MODEL_FILE)
            print("Continuous improvement training complete. Running a self–play match.")
        elif mode == 5:
            # Load and Play: just load the model without additional training.
            print("Loaded model. No additional training. Enjoy playing!")

    draw_board_pygame(board, screen)

    # Main game loop.
    while not game_over:
        # Process human input for modes 1 and 2 (when it's the human's turn).
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            if event.type == pygame.MOUSEMOTION:
                pygame.draw.rect(screen, BLACK, (0, 0, width, SQUARESIZE))
                posx = event.pos[0]
                if mode == 1 or (mode == 2 and current_player == 1):
                    color = RED if current_player == 1 else YELLOW
                    pygame.draw.circle(screen, color, (posx, int(SQUARESIZE / 2)), RADIUS)
                pygame.display.update()
            if event.type == pygame.MOUSEBUTTONDOWN:
                pygame.draw.rect(screen, BLACK, (0, 0, width, SQUARESIZE))
                if mode == 1 or (mode == 2 and current_player == 1):
                    posx = event.pos[0]
                    col = int(posx // SQUARESIZE)
                    available_actions = env.get_available_actions(board)
                    if col not in available_actions:
                        print("Invalid move. Try another column.")
                        continue
                    board, reward, game_over, info = env.step(col, current_player)
                    draw_board_pygame(board, screen)
                    if game_over:
                        pygame.time.wait(500)
                        if "winner" in info:
                            if info["winner"] == 1:
                                label = font.render("Red wins!", 1, RED)
                            elif info["winner"] == -1:
                                label = font.render("Yellow wins!", 1, YELLOW)
                            else:
                                label = font.render("Draw!", 1, WHITE)
                            screen.blit(label, (40, 10))
                            pygame.display.update()
                            time.sleep(3)
                        break
                    current_player *= -1
        # Agent moves (for modes 2, 3, 4, and 5).
        if (mode == 2 and current_player == -1) or (mode in [3, 4, 5] and not game_over):
            pygame.time.wait(500)
            state = np.array(canonical_state(board, current_player), dtype=np.float32)
            available_actions = env.get_available_actions(board)
            action = agent.select_action(state, available_actions)
            print(f"Agent ({'Red' if current_player==1 else 'Yellow'}) chooses column:", action)
            board, reward, game_over, info = env.step(action, current_player)
            draw_board_pygame(board, screen)
            if game_over:
                pygame.time.wait(500)
                if "winner" in info:
                    if info["winner"] == 1:
                        label = font.render("Red wins!", 1, RED)
                    elif info["winner"] == -1:
                        label = font.render("Yellow wins!", 1, YELLOW)
                    else:
                        label = font.render("Draw!", 1, WHITE)
                    screen.blit(label, (40, 10))
                    pygame.display.update()
                    time.sleep(3)
                break
            current_player *= -1

if __name__ == "__main__":
    main()
