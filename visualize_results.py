"""
Visualization Script for DQN and Genetic Algorithm Training Results

This script generates comprehensive graphs comparing DQN and GA performance:
- DQN: Rewards vs Episodes, Circles vs Episodes
- GA: Circles vs Generation, Learning Speed
- Comparison: GA Generations vs DQN Episodes (direct and scaled)

Usage:
    1. Place your data files (dqn_results.json, ga_results.json) in the same directory
    2. Or modify the script to load your data directly (see manual input section)
    3. Run: python visualize_results.py
    4. All graphs will be saved in the 'visual' folder

Data Format:
    DQN JSON: {"episodes": [1,2,3,...], "rewards": [...], "circles": [...]}
    GA JSON: {"generations": [1,2,3,...], "circles": [...], "forced_stop": [...]}
"""

import matplotlib.pyplot as plt
import numpy as np
import json
import os
import csv
from pathlib import Path

# Create visual folder if it doesn't exist
visual_dir = Path("visual")
visual_dir.mkdir(exist_ok=True)

def load_dqn_data(filepath=None):
    """Load DQN training data from log file or JSON/CSV"""
    # Try log file first (from main.py)
    if filepath is None:
        if os.path.exists("bench/dqn_log.txt"):
            filepath = "bench/dqn_log.txt"
        else:
            possible_files = ["dqn_results.json", "dqn_data.json", "dqn_results.csv", "dqn_data.csv"]
            for f in possible_files:
                if os.path.exists(f):
                    filepath = f
                    break
    
    if filepath and os.path.exists(filepath):
        if filepath.endswith('.txt'):  # Log file format
            episodes, rewards, circles = [], [], []
            with open(filepath, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    episodes.append(int(row['Episode']))
                    rewards.append(float(row['Reward']))
                    circles.append(int(row['Circles']))
            return {"episodes": episodes, "rewards": rewards, "circles": circles}
        elif filepath.endswith('.json'):
            with open(filepath, 'r') as f:
                data = json.load(f)
            return data
        elif filepath.endswith('.csv'):
            episodes, rewards, circles = [], [], []
            with open(filepath, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    episodes.append(int(row.get('episode', row.get('Episode', 0))))
                    rewards.append(float(row.get('reward', row.get('Reward', 0))))
                    circles.append(int(row.get('circles', row.get('Circles', 0))))
            return {"episodes": episodes, "rewards": rewards, "circles": circles}
    
    return None

def load_ga_data(filepath=None):
    """Load GA training data from log file or JSON/CSV"""
    # Try log file first (from main.py)
    if filepath is None:
        if os.path.exists("bench/ga_log.txt"):
            filepath = "bench/ga_log.txt"
        else:
            possible_files = ["ga_results.json", "ga_data.json", "ga_results.csv", "ga_data.csv"]
            for f in possible_files:
                if os.path.exists(f):
                    filepath = f
                    break
    
    if filepath and os.path.exists(filepath):
        if filepath.endswith('.txt'):  # Log file format: Gen,BestDist,AvgDist,Circles
            generations, circles = [], []
            with open(filepath, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    generations.append(int(row['Gen']))
                    circles.append(int(row['Circles']))
            
            # Last 3 entries are forced stop (as mentioned by user)
            forced_stop = [False] * len(generations)
            if len(forced_stop) >= 3:
                forced_stop[-3:] = [True, True, True]
            
            return {"generations": generations, "circles": circles, "forced_stop": forced_stop}
        elif filepath.endswith('.json'):
            with open(filepath, 'r') as f:
                data = json.load(f)
            return data
        elif filepath.endswith('.csv'):
            generations, circles, forced_stop = [], [], []
            with open(filepath, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    generations.append(int(row.get('generation', row.get('Generation', 0))))
                    circles.append(int(row.get('circles', row.get('Circles', 0))))
                    forced_stop.append(row.get('forced_stop', row.get('ForcedStop', 'False')).lower() == 'true')
            return {"generations": generations, "circles": circles, "forced_stop": forced_stop}
    
    return None

# ========== MANUAL DATA INPUT SECTION ==========
# If you have data stored in variables or need to input manually, 
# uncomment and modify the sections below:

# DQN Data Format:
# dqn_data = {
#     "episodes": [1, 2, 3, ...],  # List of episode numbers
#     "rewards": [10, 15, 20, ...],  # List of total rewards per episode
#     "circles": [0, 0, 1, ...]  # List of circles completed per episode
# }

# GA Data Format:
# ga_data = {
#     "generations": [1, 2, 3, ...],  # List of generation numbers
#     "circles": [0, 0, 1, ...],  # List of circles completed per generation
#     "forced_stop": [False, False, ..., True, True, True]  # Last 3 should be True
# }

# Load data
dqn_data = load_dqn_data()
ga_data = load_ga_data()

# If data files not found, prompt user or use sample
if dqn_data is None:
    print("=" * 60)
    print("DQN data not found in files.")
    print("Please either:")
    print("  1. Create dqn_results.json with format:")
    print('     {"episodes": [1,2,3,...], "rewards": [...], "circles": [...]}')
    print("  2. Or modify this script to add your DQN data directly (see lines 66-75)")
    print("=" * 60)
    # Uncomment below and add your DQN data:
    # dqn_data = {"episodes": [...], "rewards": [...], "circles": [...]}
    raise ValueError("DQN data not found. Please provide data.")

if ga_data is None:
    print("=" * 60)
    print("GA data not found in files.")
    print("Please either:")
    print("  1. Create ga_results.json with format:")
    print('     {"generations": [1,2,3,...], "circles": [...], "forced_stop": [...]}')
    print("  2. Or modify this script to add your GA data directly (see lines 77-82)")
    print("=" * 60)
    # Uncomment below and add your GA data:
    # ga_data = {"generations": [...], "circles": [...], "forced_stop": [...]}
    raise ValueError("GA data not found. Please provide data.")

# ==================== DQN GRAPHS ====================

# 1. DQN: Rewards vs Episodes
plt.figure(figsize=(10, 6))
plt.plot(dqn_data["episodes"], dqn_data["rewards"], linewidth=2, color='blue', alpha=0.7)
plt.xlabel('Episode', fontsize=12)
plt.ylabel('Total Reward', fontsize=12)
plt.title('DQN: Rewards Increase with Each Episode', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(visual_dir / "dqn_rewards_vs_episodes.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved: dqn_rewards_vs_episodes.png")

# 2. DQN: Circles Completed vs Episodes
plt.figure(figsize=(10, 6))
episodes = dqn_data["episodes"]
circles = dqn_data["circles"]
plt.plot(episodes, circles, linewidth=2, color='green', alpha=0.7, marker='o', markersize=3)
plt.xlabel('Episode', fontsize=12)
plt.ylabel('Circles Completed', fontsize=12)
plt.title('DQN: Circles Completed Over Episodes', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(visual_dir / "dqn_circles_vs_episodes.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved: dqn_circles_vs_episodes.png")

# ==================== GA GRAPHS ====================

# 3. GA: Circles vs Generation
plt.figure(figsize=(10, 6))
generations = ga_data["generations"]
ga_circles = ga_data["circles"]
forced_stop = ga_data.get("forced_stop", [False] * len(generations))

# Plot normal generations
normal_gen = [g for g, fs in zip(generations, forced_stop) if not fs]
normal_circles = [c for c, fs in zip(ga_circles, forced_stop) if not fs]
forced_gen = [g for g, fs in zip(generations, forced_stop) if fs]
forced_circles = [c for c, fs in zip(ga_circles, forced_stop) if fs]

if normal_gen:
    plt.plot(normal_gen, normal_circles, linewidth=2, color='purple', alpha=0.7, marker='o', markersize=4, label='Normal')
if forced_gen:
    plt.plot(forced_gen, forced_circles, linewidth=2, color='red', alpha=0.7, marker='x', markersize=6, label='Forced Stop')

plt.xlabel('Generation', fontsize=12)
plt.ylabel('Circles Completed', fontsize=12)
plt.title('GA: Circles Completed vs Generation', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(visual_dir / "ga_circles_vs_generation.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved: ga_circles_vs_generation.png")

# 4. GA: Learning Speed (Path Learning Progress)
# This shows how fast the algorithm learned - show all data but mark forced stops
plt.figure(figsize=(10, 6))
# Plot all generations to show learning progression
plt.plot(generations, ga_circles, linewidth=2, color='orange', alpha=0.7, marker='o', markersize=4, label='All Generations')
# Highlight forced stops with different markers
if forced_gen:
    plt.plot(forced_gen, forced_circles, linewidth=2, color='red', alpha=0.8, marker='x', markersize=8, label='Forced Stop', zorder=5)
plt.xlabel('Generation', fontsize=12)
plt.ylabel('Circles Completed', fontsize=12)
plt.title('GA: Learning Speed (Path Learning Progress)\n(Last 3 entries were forced stop)', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(visual_dir / "ga_learning_speed.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved: ga_learning_speed.png")

# ==================== COMPARISON GRAPHS ====================

# Prepare data for comparisons (use all generations, but we'll show forced stops separately if needed)
# For fair comparison, we'll use all GA data but can mark forced stops

# 5. Comparison: Generations vs Episodes (Direct)
plt.figure(figsize=(10, 6))
# Plot all GA generations
plt.plot(generations, ga_circles, linewidth=2, color='purple', alpha=0.7, marker='o', markersize=4, label='GA (All Generations)')
# Mark forced stops
if forced_gen:
    plt.plot(forced_gen, forced_circles, linewidth=2, color='red', alpha=0.8, marker='x', markersize=6, label='GA (Forced Stop)', zorder=5)
plt.plot(dqn_data["episodes"], dqn_data["circles"], linewidth=2, color='blue', alpha=0.7, marker='s', markersize=3, label='DQN (Episodes)')
plt.xlabel('Generation / Episode', fontsize=12)
plt.ylabel('Circles Completed', fontsize=12)
plt.title('Comparison: GA Generations vs DQN Episodes', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(visual_dir / "comparison_generations_vs_episodes.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved: comparison_generations_vs_episodes.png")

# 6. Comparison: 50 * Generations vs Episodes (Scaled)
# Each generation is like 50 episodes
ga_episodes_equivalent = [g * 50 for g in generations]
forced_episodes_equivalent = [g * 50 for g in forced_gen] if forced_gen else []

plt.figure(figsize=(10, 6))
plt.plot(ga_episodes_equivalent, ga_circles, linewidth=2, color='purple', alpha=0.7, marker='o', markersize=4, label='GA (50 × Generations)')
if forced_episodes_equivalent:
    plt.plot(forced_episodes_equivalent, forced_circles, linewidth=2, color='red', alpha=0.8, marker='x', markersize=6, label='GA (Forced Stop)', zorder=5)
plt.plot(dqn_data["episodes"], dqn_data["circles"], linewidth=2, color='blue', alpha=0.7, marker='s', markersize=3, label='DQN (Episodes)')
plt.xlabel('Episode Equivalent', fontsize=12)
plt.ylabel('Circles Completed', fontsize=12)
plt.title('Comparison: GA (50× Generations) vs DQN Episodes', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(visual_dir / "comparison_scaled_generations_vs_episodes.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved: comparison_scaled_generations_vs_episodes.png")

print(f"\nAll graphs saved to '{visual_dir}' folder!")

