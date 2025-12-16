import matplotlib.pyplot as plt
import pandas as pd
import json
from pathlib import Path

# Create visual folder if it doesn't exist
visual_dir = Path("visual")
visual_dir.mkdir(exist_ok=True)

def load_dqn_data(filepath=None):
    """Load DQN training data from log file or JSON/CSV"""
    # Try log file first (from main.py)
    if filepath is None:
        if Path("bench/dqn_log.txt").exists():
            filepath = "bench/dqn_log.txt"
        else:
            possible_files = ["dqn_results.json", "dqn_data.json", "dqn_results.csv", "dqn_data.csv"]
            for f in possible_files:
                if Path(f).exists():
                    filepath = f
                    break
    
    if filepath and Path(filepath).exists():
        if filepath.endswith('.json'):
            with open(filepath, 'r') as f:
                return json.load(f)
        
        # Handle .txt (log) and .csv using pandas
        try:
            df = pd.read_csv(filepath)
            # Normalize columns to lowercase to handle 'Episode' vs 'episode' automatically
            df.columns = df.columns.str.strip().str.lower()
            
            # Map columns based on expected names
            ep_col = 'episode' if 'episode' in df.columns else df.columns[0]
            rew_col = 'reward' if 'reward' in df.columns else df.columns[1]
            circ_col = 'circles' if 'circles' in df.columns else df.columns[2]

            return {
                "episodes": df[ep_col].tolist(),
                "rewards": df[rew_col].tolist(),
                "circles": df[circ_col].tolist()
            }
        except Exception as e:
            print(f"Error reading DQN file: {e}")
            return None
    
    return None

def load_ga_data(filepath=None):
    """Load GA training data from log file or JSON/CSV"""
    # Try log file first (from main.py)
    if filepath is None:
        if Path("bench/ga_log.txt").exists():
            filepath = "bench/ga_log.txt"
        else:
            possible_files = ["ga_results.json", "ga_data.json", "ga_results.csv", "ga_data.csv"]
            for f in possible_files:
                if Path(f).exists():
                    filepath = f
                    break
    
    if filepath and Path(filepath).exists():
        if filepath.endswith('.json'):
            with open(filepath, 'r') as f:
                return json.load(f)
        
        # Handle .txt (log) and .csv using pandas
        try:
            df = pd.read_csv(filepath)
            df.columns = df.columns.str.strip().str.lower()
            
            # Determine generation column name (log uses 'gen', csv might use 'generation')
            gen_col = 'gen' if 'gen' in df.columns else 'generation'
            if gen_col not in df.columns:
                 # Fallback if header is completely different, though unlikely based on spec
                 gen_col = df.columns[0]

            generations = df[gen_col].tolist()
            circles = df['circles'].tolist()

            # Handle forced_stop logic
            if 'forced_stop' in df.columns or 'forcedstop' in df.columns:
                col = 'forced_stop' if 'forced_stop' in df.columns else 'forcedstop'
                forced_stop = df[col].astype(str).str.lower() == 'true'
                forced_stop = forced_stop.tolist()
            else:
                # Last 3 entries are forced stop (as mentioned by user)
                forced_stop = [False] * len(generations)
                if len(forced_stop) >= 3:
                    forced_stop[-3:] = [True, True, True]
            
            return {"generations": generations, "circles": circles, "forced_stop": forced_stop}
        except Exception as e:
            print(f"Error reading GA file: {e}")
            return None
    
    return None


dqn_data = load_dqn_data()
ga_data = load_ga_data()

# If data files not found, prompt user or use sample
if dqn_data is None:
    print("=" * 60)
    print("DQN data not found in files.")
    print("Please either:")
    print("  1. Create dqn_results.json with format:")
    print('     {"episodes": [1,2,3,...], "rewards": [...], "circles": [...]}')
    print("  2. Or modify this script to add your DQN data directly")
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
    print("  2. Or modify this script to add your GA data directly")
    print("=" * 60)
    # Uncomment below and add your GA data:
    # ga_data = {"generations": [...], "circles": [...], "forced_stop": [...]}
    raise ValueError("GA data not found. Please provide data.")


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

# COMPARISION GRAPHS


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