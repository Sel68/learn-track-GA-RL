# Autonomous Navigation: Genetic Algorithm vs. Deep Q-Network

A comparative study implementing **Evolutionary Computation (GA)** and **Reinforcement Learning (DQN)** to train an autonomous vehicle to navigate a track using sensor-based perception.

## ⚡ Quick Start

### Prerequisites
* Python 3.7+
* `torch`, `numpy`, `matplotlib`

### Installation & Run
```bash
# 1. Clone the repository
git clone <repository-url>
cd learn-track-GA-RL

# 2. Install dependencies
pip install torch numpy matplotlib

# 3. Run the simulation
# Note: Edit 'MODE' in main.py to switch between "DQN" and "GA"
python main.py

# 4. Generate performance graphs (after training)
python visualize_results.py