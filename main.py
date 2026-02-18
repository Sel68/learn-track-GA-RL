import numpy as np
import torch
import math
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import random
from collections import deque
import os
from track_setup import create_track, draw_track

CONFIG = {
    "MODE": "DQN",  # "GA" or "DQN"
    "TRACK_SIZE": 100,
    "TRACK_WIDTH": 20,
    "TRACK_N_SIDES": 4,  # Number of sides for the polygon track (4=square, 3=triangle, 5=pentagon, etc.)
    "N_SENSORS": 8,
    "SENSOR_RANGE": 30,
    "FPS": 60, #Speed/smoothness of sim
    "RENDER_EVERY": 10, # Frames render for every frame shown

    # GA Hyperparameters
    "GA_POP_SIZE": 30,
    "GA_ELITISM": 0.4, # Top % survival
    "GA_MUTATION_RATE": 0.6,
    "GA_SIGMA": 0.2, # Gaussian noise std dev

    # DQN Hyperparameters
    "DQN_GAMMA": 0.99,
    "DQN_EPS_START": 1.0,
    "DQN_EPS_END": 0.05,
    "DQN_EPS_DECAY": 2000,
    "DQN_LR": 1e-3,
    "DQN_BATCH_SIZE": 32,
    "DQN_MEMORY_SIZE": 100000,
    "DQN_TARGET_UPDATE": 1000,  # Hard update every N steps
    "DQN_Hidden": 128,
    "DQN_LEARNING_START": 1000,  # Start training after this many steps
    "DQN_UPDATE_FREQ": 4  # Train every N steps
}

# if cuda avail
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Track is now created using track_setup.create_track()

class Car:
    def __init__(self, track):
        self.track = track
        self.reset()
        
    def reset(self):
        # Start at first checkpoint, facing towards next checkpoint
        checkpoint = self.track.checkpoints[0]
        self.x = checkpoint[0]
        self.y = checkpoint[1]
        self.angle = 0 # Radians
        self.speed = 2.0 
        self.alive = True
        self.distance_traveled = 0
        self.time_alive = 0
        self.circles = 0
        self.current_checkpoint = 0
        self.radars = np.zeros(CONFIG["N_SENSORS"])
        return self.get_state()

    def step(self, action):
        if not self.alive:
            return self.get_state(), 0, True, {}

        # DQN Action Handling
        steering = 0
        if isinstance(action, (list, np.ndarray, torch.Tensor)):
            steering = float(action[0]) * 0.2
        else:
            # Discrete Actions: 0=Left, 1=Straight, 2=Right
            if action == 0: steering = -0.2 
            elif action == 2: steering = 0.2

        # Physics
        self.angle += steering
        self.x += math.cos(self.angle) * self.speed
        self.y += math.sin(self.angle) * self.speed
        
        self.time_alive += 1
        self.distance_traveled += self.speed
        
        reward = 0

        # upd sensors
        self.radars = self._sense()

        if self.track.check_collision(self.x, self.y):
            self.alive = False
            reward = -10 # lowered penalty to still motivate progress
        else:
            '''
            marginalised reward for survival now. Was moving in small circles. 
            '''
            reward += 0.5
            
            #added reward for getting closer to walls
            # min_wall_dist = np.min(self.radars)
            # reward += min_wall_dist * 0.5 

            # Checkpoints
            cx, cy = self.track.checkpoints[self.current_checkpoint]
            dist_to_cp = math.sqrt((self.x - cx)**2 + (self.y - cy)**2)
            if dist_to_cp < CONFIG["TRACK_WIDTH"]:
                self.current_checkpoint = (self.current_checkpoint + 1) % self.track.n_sides
                reward += 10 
                self.last_cp_time = self.time_alive
                if self.current_checkpoint == 0:
                    self.circles += 1
                    reward += 20
            
            if self.time_alive - self.last_cp_time > 200:
                self.alive = False
                reward = -50
        
        return self.get_state(), reward, not self.alive, {}

    def _sense(self):
        radars = []
        # Spread sensors over 180 degrees
        start_angle = self.angle - math.pi / 2
        step_angle = math.pi / (CONFIG["N_SENSORS"] - 1)
        
        for i in range(CONFIG["N_SENSORS"]):
            ray_angle = start_angle + i * step_angle
            ray_dir = (math.cos(ray_angle), math.sin(ray_angle))
            dist = self.track.get_ray_intersection((self.x, self.y), ray_dir, CONFIG["SENSOR_RANGE"])
            # Normalize 0-1
            radars.append(dist / CONFIG["SENSOR_RANGE"]) 
        return np.array(radars)

    def get_state(self):
        return torch.FloatTensor(self.radars).to(device)

class EvolutionNet(nn.Module):
    def __init__(self):
        super(EvolutionNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(CONFIG["N_SENSORS"], 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 2), # Steering, Accel
            nn.Tanh() # Output -1 to 1
        )

    def forward(self, x):
        return self.fc(x)
    

#GA
class GeneticPopulation:
    def __init__(self, log_path: str = "bench/ga_log.txt"):
        self.population = [EvolutionNet().to(device) for _ in range(CONFIG["GA_POP_SIZE"])]
        self.gen_count = 0
        
        # Logger Setup (per-run log file)
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.log_path = log_path
        with open(self.log_path, "w") as f:
            f.write("Gen,BestDist,AvgDist,Circles\n")

    def evaluate(self, track):
        scores = []
        for net in self.population:
            car = Car(track)
            state = car.get_state()
            # Simulation loop for one car
            while car.alive and car.time_alive < 10000: # Timeout limit
                with torch.no_grad():
                    action = net(state).cpu().numpy()
                state, _, done, _ = car.step(action)
            scores.append((car.distance_traveled, car.circles, net))
        
        return scores

    def evolve(self, scored_pop):
        # Sort by distance, descending
        scored_pop.sort(key=lambda x: x[0], reverse=True)
        
        # Logging
        best_dist = scored_pop[0][0]
        avg_dist = sum(s[0] for s in scored_pop) / len(scored_pop)
        best_circles = scored_pop[0][1]
        print(f"GA Gen {self.gen_count}: Best Dist: {best_dist:.2f}, Avg: {avg_dist:.2f}")
        with open(self.log_path, "a") as f:
            f.write(f"{self.gen_count},{best_dist},{avg_dist},{best_circles}\n")

        # Selection (Elitism)
        retain_len = int(len(scored_pop) * CONFIG["GA_ELITISM"])
        new_pop = [s[2] for s in scored_pop[:retain_len]]
        
        # Mutation & Crossover
        while len(new_pop) < CONFIG["GA_POP_SIZE"]:
            parent = random.choice(new_pop[:retain_len])
            child = EvolutionNet().to(device)
            child.load_state_dict(parent.state_dict())
            
            #Gaussian Noise for exploration
            with torch.no_grad():
                for param in child.parameters():
                    noise = torch.randn_like(param) * CONFIG["GA_SIGMA"]
                    param.add_(noise)
            new_pop.append(child)
        
        self.population = new_pop
        self.gen_count += 1
        # Returns only the best model for visualization
        return scored_pop[0][2] 

class DQNNet(nn.Module):
    def __init__(self, inputs, outputs):
        super(DQNNet, self).__init__()
        # Dueling architecture: separate value and advantage streams
        self.fc = nn.Sequential(
            nn.Linear(inputs, CONFIG["DQN_Hidden"]),
            nn.ReLU(),
            nn.Linear(CONFIG["DQN_Hidden"], CONFIG["DQN_Hidden"]),
            nn.ReLU()
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(CONFIG["DQN_Hidden"], CONFIG["DQN_Hidden"]),
            nn.ReLU(),
            nn.Linear(CONFIG["DQN_Hidden"], 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(CONFIG["DQN_Hidden"], CONFIG["DQN_Hidden"]),
            nn.ReLU(),
            nn.Linear(CONFIG["DQN_Hidden"], outputs)
        )
        
        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        features = self.fc(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        # Dueling formula: Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        return value + advantage - advantage.mean(dim=1, keepdim=True)
    
class ReplayBuffer:
    def __init__(self, capacity, state_dim):
        self.capacity = capacity
        self.state_dim = state_dim
        self.position = 0
        self.filled = False
        
        # Pre-allocate numpy arrays for efficient storage
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
    
    def push(self, state, action, reward, next_state, done):
        # Convert tensors to numpy if needed
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
        if isinstance(next_state, torch.Tensor):
            next_state = next_state.cpu().numpy()
        
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = float(done)
        
        self.position = (self.position + 1) % self.capacity
        if self.position == 0:
            self.filled = True
    
    def sample(self, batch_size):
        if self.filled:
            max_idx = self.capacity
        else:
            max_idx = self.position
        
        indices = np.random.randint(0, max_idx, size=batch_size)
        
        # Convert to tensors on device
        states = torch.FloatTensor(self.states[indices]).to(device)
        actions = torch.LongTensor(self.actions[indices]).to(device)
        rewards = torch.FloatTensor(self.rewards[indices]).to(device)
        next_states = torch.FloatTensor(self.next_states[indices]).to(device)
        dones = torch.FloatTensor(self.dones[indices]).to(device)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return self.capacity if self.filled else self.position

class DQNAgent:
    def __init__(self, log_path: str = "bench/dqn_log.txt"):
        self.policy_net = DQNNet(CONFIG["N_SENSORS"], 3).to(device)
        self.target_net = DQNNet(CONFIG["N_SENSORS"], 3).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=CONFIG["DQN_LR"])
        self.memory = ReplayBuffer(CONFIG["DQN_MEMORY_SIZE"], CONFIG["N_SENSORS"])
        self.steps_done = 0
        self.episode = 0
        self.update_counter = 0
        
        # Logs
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.log_path = log_path
        if os.path.exists(self.log_path): 
            os.remove(self.log_path)
        with open(self.log_path, "w") as f:
            f.write("Episode,Reward,Duration,Epsilon,Circles\n")

    def select_action(self, state):
        """Epsilon-greedy action selection"""
        eps = CONFIG["DQN_EPS_END"] + (CONFIG["DQN_EPS_START"] - CONFIG["DQN_EPS_END"]) * \
              math.exp(-1.0 * self.steps_done / CONFIG["DQN_EPS_DECAY"])
        self.steps_done += 1
        
        if random.random() > eps:
            with torch.no_grad():
                return self.policy_net(state.unsqueeze(0)).argmax(dim=1).item()
        else:
            return random.randint(0, 2)

    def optimize_model(self):
        """Optimize model using Double DQN with periodic target updates"""
        if len(self.memory) < CONFIG["DQN_LEARNING_START"]:
            return
        
        # Only train every N steps
        if self.update_counter % CONFIG["DQN_UPDATE_FREQ"] != 0:
            self.update_counter += 1
            return
        self.update_counter += 1
        
        if len(self.memory) < CONFIG["DQN_BATCH_SIZE"]:
            return
        
        # Sample from replay buffer
        states, actions, rewards, next_states, dones = self.memory.sample(CONFIG["DQN_BATCH_SIZE"])
        
        # Current Q-values
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Double DQN: use policy net to select action, target net to evaluate
        with torch.no_grad():
            # Policy net chooses best action for next state
            next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            # Target net evaluates that action
            next_q_values = self.target_net(next_states).gather(1, next_actions)
            # Bellman equation
            target_q_values = rewards.unsqueeze(1) + CONFIG["DQN_GAMMA"] * next_q_values * (1 - dones.unsqueeze(1))
        
        # Huber loss
        loss = nn.SmoothL1Loss()(q_values, target_q_values)
        
        # Optimization step
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Hard update target network periodically
        if self.steps_done % CONFIG["DQN_TARGET_UPDATE"] == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def log_episode(self, total_reward, duration, circles):
        eps = CONFIG["DQN_EPS_END"] + (CONFIG["DQN_EPS_START"] - CONFIG["DQN_EPS_END"]) * \
              math.exp(-1.0 * self.steps_done / CONFIG["DQN_EPS_DECAY"])
        
        print(f"DQN Ep {self.episode}: Rew {total_reward:.1f} | Steps {duration} | Circles {circles} | Eps {eps:.2f}")
        with open(self.log_path, "a") as f:
            f.write(f"{self.episode},{total_reward},{duration},{eps:.2f},{circles}\n")
        self.episode += 1

#Visualisation is now handled by track_setup.draw_track()

def run_simulation():
    # Create track with specified number of sides
    track = create_track(CONFIG["TRACK_SIZE"], CONFIG["TRACK_WIDTH"], CONFIG["TRACK_N_SIDES"])
    
    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 6))
    
    if CONFIG["MODE"] == "GA":
        print("Starting Genetic Algorithm...")
        ga = GeneticPopulation()
        
        while True:
            # Fast, no render
            scored_pop = ga.evaluate(track)
            
            #Visualize Best Agent of Generation
            best_net = scored_pop[0][2]
            demo_car = Car(track)
            state = demo_car.get_state()
            
            while demo_car.alive and demo_car.time_alive < 10000:
                with torch.no_grad():
                    action = best_net(state).cpu().numpy()
                state, _, done, _ = demo_car.step(action)
                
                if demo_car.time_alive % CONFIG["RENDER_EVERY"] == 0:
                    draw_track(ax, track, demo_car, demo_car.radars, CONFIG)
                    plt.title(f"GA Gen {ga.gen_count} | Dist: {demo_car.distance_traveled:.1f}")
                    plt.pause(1/CONFIG["FPS"])
            # Evolve
            ga.evolve(scored_pop)

    elif CONFIG["MODE"] == "DQN":
        print("Starting Deep Q-Network...")
        agent = DQNAgent()
        
        # Frame skipping constant
        FRAME_SKIP = 4 
        
        while True:
            car = Car(track)
            state = car.get_state().cpu().numpy() if isinstance(car.get_state(), torch.Tensor) else car.get_state()
            total_reward = 0
            
            while car.alive and car.time_alive < 10000:
                # Select action
                action = agent.select_action(torch.FloatTensor(state).to(device))
                
                # Repeat action for stability
                reward_accum = 0
                for _ in range(FRAME_SKIP):
                    next_state, r, done, _ = car.step(action)
                    next_state_np = next_state.cpu().numpy() if isinstance(next_state, torch.Tensor) else next_state
                    reward_accum += r
                    if done:
                        break
                
                # Store transition
                agent.memory.push(state, action, reward_accum, next_state_np, done)
                state = next_state_np
                total_reward += reward_accum
                
                # Train
                agent.optimize_model()
                
                # Visualize every N episodes
                if agent.episode % 10 == 0 and car.time_alive % CONFIG["RENDER_EVERY"] == 0:
                    draw_track(ax, track, car, car.radars, CONFIG)
                    plt.title(f"DQN Ep {agent.episode} | Rew: {total_reward:.1f}")
                    plt.pause(1/CONFIG["FPS"])
            
            agent.log_episode(total_reward, car.time_alive, car.circles)

if __name__ == "__main__":
    run_simulation()