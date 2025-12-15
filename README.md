# Pong RL

This project is a Reinforcement Learning implementation where two AI agents learn to play Pong against each other. It uses **PyTorch** for the neural networks and **Pygame** for the game environment. The agents utilize the **Deep Q-Network (DQN)** algorithm to master the game through self-play.

Instead of using raw pixels, the agents perceive the environment through a state vector (ball position, velocity, and paddle positions), allowing for faster and more efficient training.

## Key Features

* **Deep Q-Learning (DQN):** Agents learn to maximize future rewards using a neural network.
* **Self-Play:** Two agents (Left and Right) train simultaneously by playing against one another.
* **Experience Replay:** Stores past experiences in a buffer and samples them randomly to stabilize training.
* **Target Network:** Uses a separate target network that updates periodically to prevent oscillation during learning.
* **Customizable Environment:** Hyperparameters like learning rate, game speed, and epsilon decay can be easily adjusted in `config.py`.
* **Save & Load System:** Models are saved automatically during training and can be loaded later for testing/watching.

## Project Structure

To ensure the imports work correctly, please organize your files into the following directory structure:

```text
pong-rl/
├── ai/
│   ├── agent.py        # Agent logic and training loop
│   ├── memory.py       # Replay Buffer implementation
│   └── model.py        # Neural Network architecture (DQN)
├── core/
│   ├── env.py          # Pong game environment
│   └── entities.py     # Game objects (Ball, Paddle)
├── models/             # Directory for saved models (.pth)
├── logs/               # Directory for training logs
├── config.py           # Configuration and hyperparameters
├── main.py             # Entry point (Train/Test)
└── environment.yml     # Conda environment file
```

## Installation

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/ozgrakby/pong-rl.git
    cd pong-rl
    ```

2.  **Create the Environment:**
    It is recommended to use `conda` to manage dependencies.
    ```bash
    conda env create -f environment.yml
    conda activate pong-rl
    ```

3.  **Install Missing Dependencies:**
    If `pygame` is not installed automatically, you can install it via pip:
    ```bash
    pip install pygame
    ```

## Usage

Run the main script to start the application:

```bash
python main.py
```

You will be presented with two options:

### 1. Train Models
* Press `1` to start training.
* The game runs without rendering (headless) to maximize training speed.
* Progress is printed to the terminal and saved to the `logs/` folder.
* Models are saved to the `models/` directory at intervals specified in `config.py`.
* You can stop training at any time using `Ctrl+C`; the current state will be saved.

### 2. Watch Models (Test)
* Press `2` to watch the trained agents play.
* This mode renders the game window so you can see the AI in action.
* **Requirement:** You must have `agent_left.pth` and `agent_right.pth` inside the `models/` directory.

## Configuration

You can tweak the training and game parameters in `config.py`:

* **Training Params:**
    * `LR`: Learning Rate.
    * `BATCH_SIZE`: Number of samples taken from memory.
    * `GAMMA`: Discount factor for future rewards.
    * `EPSILON_START` / `EPSILON_DECAY`: Controls the exploration vs. exploitation trade-off.
* **Game Params:**
    * `FPS`: Controls the speed of the game (mostly for the Test mode).
    * `BALL_MAX_SPEED`: Limits how fast the ball can travel.

## Technical Details

* **State Space (Input):** A normalized vector of size 6:
    * `[Ball X, Ball Y, Ball Velocity X, Ball Velocity Y, Left Paddle Y, Right Paddle Y]`
* **Action Space (Output):** 3 discrete actions:
    * `0`: Stay
    * `1`: Move Up
    * `2`: Move Down
* **Reward System:**
    * `+0.1`: Successfully hitting the ball.
    * `+1.0`: Scoring a goal.
    * `-1.0`: Conceding a goal.

---

**Note:** Reinforcement Learning takes time. It might take several thousand episodes before the agents start hitting the ball consistently. Be patient!