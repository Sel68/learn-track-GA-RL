# Learn Track: Genetic Algorithm vs Deep Q-Network

A comparative study of two reinforcement learning approaches for autonomous car navigation: **Genetic Algorithm (GA)** and **Deep Q-Network (DQN)**. This project implements both algorithms to train a car to navigate a square racetrack, learning to avoid walls and complete laps around the track.

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
  - [Environment](#environment)
  - [Genetic Algorithm](#genetic-algorithm)
  - [Deep Q-Network](#deep-q-network)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Visualization](#visualization)
- [Results](#results)

## 🎯 Overview

This project compares two different machine learning approaches for training an autonomous car:

1. **Genetic Algorithm (GA)**: An evolutionary approach that evolves neural network weights through selection, mutation, and crossover operations across generations.

2. **Deep Q-Network (DQN)**: A reinforcement learning approach that uses a deep neural network to approximate Q-values, learning optimal actions through trial and error with experience replay.

Both algorithms train a car to navigate a square racetrack with inner and outer walls, using sensor-based perception to avoid collisions and complete laps.

## 📁 Project Structure

```
learn-track-GA-RL/
├── main.py                 # Main simulation and training code
├── visualize_results.py    # Visualization script for training results
├── bench/                  # Training logs directory
│   ├── dqn_log.txt         # DQN training logs
│   └── ga_log.txt          # GA training logs
└── visual/                 # Generated visualization graphs
    ├── dqn_rewards_vs_episodes.png
    ├── dqn_circles_vs_episodes.png
    ├── ga_circles_vs_generation.png
    ├── ga_learning_speed.png
    ├── comparison_generations_vs_episodes.png
    └── comparison_scaled_generations_vs_episodes.png
```

## 🔧 How It Works

### Environment

The simulation environment consists of:

#### **Track**
- **Square racetrack** with outer walls (0 to 100 units) and inner walls (20 to 80 units)
- **Track width**: 20 units (the drivable area between inner and outer walls)
- **Checkpoints**: 4 checkpoints positioned at the center of each side (bottom, right, top, left)
- **Collision detection**: Car has a radius of 3.5 units; collision occurs if the car touches any wall

#### **Car**
- **Starting position**: Bottom center of the track, facing right
- **Physics**: 
  - Constant speed: 2.0 units per frame
  - Steering angle changes based on actions
  - Position updated using trigonometry: `x += cos(angle) * speed`, `y += sin(angle) * speed`
- **Sensors**: 8 raycast sensors spread over 180 degrees in front of the car
  - Each sensor detects distance to nearest wall (normalized 0-1)
  - Sensor range: 30 units
  - Sensors provide the state representation for both algorithms

#### **Reward System**
- **Survival reward**: +0.1 per frame (encourages staying alive)
- **Wall proximity reward**: +0.5 × minimum wall distance (encourages staying centered in track)
- **Checkpoint reward**: +10 when reaching a checkpoint
- **Lap completion reward**: +20 when completing a full lap (returning to first checkpoint)
- **Collision penalty**: -15 when hitting a wall

### Genetic Algorithm

The GA approach evolves a population of neural networks over generations.

#### **Network Architecture**
- **Input**: 8 sensor readings (normalized distances)
- **Hidden layers**: 16 → 16 neurons with ReLU activation
- **Output**: 2 continuous values (steering angle, acceleration) with Tanh activation (-1 to 1)
- **Steering**: Output[0] × 0.1 radians per frame

#### **Evolution Process**

1. **Initialization**: Create a population of random neural networks (default: 5 networks)

2. **Evaluation**:
   - Each network controls a car for one simulation run
   - Car runs until collision or timeout (10,000 frames)
   - Fitness = distance traveled (primary metric)
   - Secondary metrics: circles completed, time alive

3. **Selection (Elitism)**:
   - Keep top 20% of population (best performers)
   - These elite networks survive to next generation

4. **Reproduction**:
   - Fill remaining population slots by:
     - Selecting a random parent from elite group
     - Creating a child network with identical weights
     - Applying Gaussian noise mutation: `weight += N(0, σ²)` where σ = 0.3
   - This creates variation while preserving good traits

5. **Repeat**: Process continues indefinitely, with each generation potentially improving

#### **Key Features**
- **No gradient computation**: Pure evolutionary search, no backpropagation
- **Population-based**: Multiple solutions explored simultaneously
- **Mutation-driven exploration**: Gaussian noise introduces diversity
- **Elitism**: Best solutions preserved across generations

### Deep Q-Network

The DQN approach uses reinforcement learning with experience replay and target networks.

#### **Network Architecture**
- **Input**: 8 sensor readings (normalized distances)
- **Hidden layers**: 64 → 64 neurons with ReLU activation
- **Output**: 3 Q-values (one for each discrete action)
- **Actions**: 
  - 0: Turn left (-0.15 radians)
  - 1: Go straight (0 radians)
  - 2: Turn right (+0.15 radians)

#### **Learning Process**

1. **Action Selection (ε-greedy)**:
   - With probability ε: random action (exploration)
   - With probability (1-ε): action with highest Q-value (exploitation)
   - ε decays from 1.0 to 0.05 over 1000 steps

2. **Experience Replay**:
   - Store transitions (state, action, reward, next_state, done) in replay buffer
   - Buffer capacity: 50,000 transitions
   - Sample random batches of 128 transitions for training
   - Breaks correlation between consecutive experiences

3. **Q-Learning Update**:
   - **Target Q-value**: `Q_target = reward + γ × max(Q(next_state)) × (1 - done)`
   - **Loss**: Smooth L1 Loss between predicted Q-value and target
   - **Gradient clipping**: Clips gradients to max norm of 10 (prevents exploding gradients)

4. **Target Network**:
   - Separate target network with identical architecture
   - Updated every 1000 steps by copying policy network weights
   - Provides stable targets for Q-learning updates
   - Uses soft update: `target_net = policy_net` (hard copy)

5. **Frame Skipping**:
   - Action repeated for 4 frames for stability
   - Accumulates rewards over skipped frames

#### **Key Features**
- **Off-policy learning**: Learns optimal policy while following ε-greedy policy
- **Experience replay**: Reuses past experiences for efficient learning
- **Target network**: Stabilizes training by providing fixed targets
- **Discrete actions**: Simpler action space than continuous GA output

## 🚀 Installation

### Prerequisites

- Python 3.7+
- PyTorch
- NumPy
- Matplotlib

### Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd learn-track-GA-RL
   ```

2. **Install dependencies**:
   ```bash
   pip install torch numpy matplotlib
   ```

   Or create a `requirements.txt`:
   ```txt
   torch>=1.9.0
   numpy>=1.21.0
   matplotlib>=3.4.0
   ```

   Then install:
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Usage

### Running the Simulation

1. **Edit configuration** in `main.py`:
   ```python
   CONFIG = {
       "MODE": "DQN",  # or "GA"
       # ... other parameters
   }
   ```

2. **Run the simulation**:
   ```bash
   python main.py
   ```

3. **Watch the training**:
   - The simulation will display a real-time visualization
   - Training logs are saved to `bench/dqn_log.txt` or `bench/ga_log.txt`
   - Press Ctrl+C to stop training

### Generating Visualizations

After training, generate comparison graphs:

```bash
python visualize_results.py
```

This will:
- Load training logs from `bench/` directory
- Generate 6 visualization graphs in the `visual/` folder
- Display progress messages

## ⚙️ Configuration

All configuration parameters are in the `CONFIG` dictionary in `main.py`:

### Environment Parameters
- `TRACK_SIZE`: Size of the square track (default: 100)
- `TRACK_WIDTH`: Width of the drivable area (default: 20)
- `N_SENSORS`: Number of raycast sensors (default: 8)
- `SENSOR_RANGE`: Maximum detection range (default: 30)
- `FPS`: Simulation frames per second for visualization (default: 60)
- `RENDER_EVERY`: Render every Nth frame (default: 100, for faster training)

### Genetic Algorithm Parameters
- `GA_POP_SIZE`: Population size (default: 5)
- `GA_ELITISM`: Fraction of top performers to keep (default: 0.2 = 20%)
- `GA_MUTATION_RATE`: Probability of mutation (default: 0.6, though always applied)
- `GA_SIGMA`: Standard deviation of Gaussian mutation noise (default: 0.3)

### Deep Q-Network Parameters
- `DQN_GAMMA`: Discount factor for future rewards (default: 0.99)
- `DQN_EPS_START`: Initial exploration rate (default: 1.0)
- `DQN_EPS_END`: Final exploration rate (default: 0.05)
- `DQN_EPS_DECAY`: Steps for ε decay (default: 1000)
- `DQN_LR`: Learning rate (default: 1e-3)
- `DQN_BATCH_SIZE`: Batch size for training (default: 128)
- `DQN_MEMORY_SIZE`: Replay buffer capacity (default: 50000)
- `DQN_TARGET_UPDATE`: Steps between target network updates (default: 10, but code uses 1000)
- `DQN_Hidden`: Hidden layer size (default: 64)
- `DQN_TAU`: Soft update coefficient (default: 0.005, but code uses hard updates)

## 📊 Visualization

The `visualize_results.py` script generates comprehensive comparison graphs:

### DQN Graphs
1. **Rewards vs Episodes**: Shows learning progress through total reward per episode
2. **Circles vs Episodes**: Tracks lap completion over training

### GA Graphs
3. **Circles vs Generation**: Shows best agent's lap completion per generation
4. **Learning Speed**: Path learning progress (excluding forced stop entries)

### Comparison Graphs
5. **Direct Comparison**: GA generations vs DQN episodes (raw comparison)
6. **Scaled Comparison**: GA generations × 50 vs DQN episodes (accounting for computational cost)

### Log File Format

**DQN Log** (`bench/dqn_log.txt`):
```
Episode,Reward,Duration,Epsilon,Circles
0,15.2,150,0.95,0
1,18.5,180,0.92,0
...
```

**GA Log** (`bench/ga_log.txt`):
```
Gen,BestDist,AvgDist,Circles
0,125.5,98.3,0
1,156.2,112.4,0
...
```

## 📈 Results

The project tracks several performance metrics:

- **Distance Traveled**: Primary fitness metric for GA, secondary for DQN
- **Circles Completed**: Number of full laps around the track
- **Time Alive**: Frames before collision or timeout
- **Total Reward**: Cumulative reward per episode (DQN) or generation (GA)

### Expected Behavior

- **Early Training**: Both algorithms explore randomly, frequent collisions
- **Mid Training**: Agents learn to avoid walls, complete partial laps
- **Late Training**: Agents consistently complete full laps, optimize path

### Performance Comparison

- **GA**: Typically learns faster in terms of generations (each generation evaluates entire population)
- **DQN**: More sample-efficient per episode, but requires more episodes
- **Computational Cost**: GA generation ≈ 50 DQN episodes (population size × evaluation time)

## 🔍 Technical Details

### State Representation
- **8-dimensional vector**: Normalized sensor distances (0-1)
- Sensors spread 180° in front of car
- Provides local perception of track boundaries

### Action Space
- **GA**: Continuous steering angle (output × 0.1 radians)
- **DQN**: Discrete actions (left, straight, right)

### Reward Shaping
- Designed to balance exploration and exploitation
- Survival reward prevents premature termination
- Wall proximity reward encourages center-line driving
- Checkpoint rewards provide intermediate goals

### Collision Detection
- Uses geometric collision detection with car radius
- Checks outer bounds and inner obstacle bounds
- Efficient O(1) collision check per frame

## 🛠️ Customization

### Adding New Features
- **New sensors**: Modify `Car._sense()` method
- **Different track shapes**: Modify `Track.__init__()` to define new wall geometry
- **New reward functions**: Modify reward calculation in `Car.step()`
- **Different network architectures**: Modify `EvolutionNet` or `DQNNet` classes

### Experimentation Tips
- Start with smaller populations/episodes for faster iteration
- Adjust mutation rate (GA) or learning rate (DQN) for different learning speeds
- Modify reward weights to emphasize different behaviors
- Change sensor count/range to test perception sensitivity

## 📝 Notes

- The simulation runs indefinitely until manually stopped
- Training logs are appended to files (clear files to start fresh)
- Visualization requires training logs to be present
- GPU acceleration is used if CUDA is available (PyTorch)

## 🤝 Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## 📄 License

This project is open source and available for educational and research purposes.
