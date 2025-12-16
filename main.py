import numpy as np
import torch
import math
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import random
from collections import deque
import os

CONFIG = {
    "MODE": "GA",  # "GA" or "DQN"
    "TRACK_SIZE": 100,
    "TRACK_WIDTH": 20,
    "N_SENSORS": 8,
    "SENSOR_RANGE": 30,
    "FPS": 60, #Speed/smoothness of sim
    "RENDER_EVERY": 1, # Frames render for every frame shown

    # GA Hyperparameters
    "GA_POP_SIZE": 50,
    "GA_ELITISM": 0.4, # Top % survival
    "GA_MUTATION_RATE": 0.6,
    "GA_SIGMA": 0.2, # Gaussian noise std dev

    # DQN Hyperparameters
    "DQN_GAMMA": 0.99,
    "DQN_EPS_START": 1.0,
    "DQN_EPS_END": 0.05,
    "DQN_EPS_DECAY": 500,
    "DQN_LR": 5e-4,
    "DQN_BATCH_SIZE": 128,
    "DQN_MEMORY_SIZE": 50000,
    "DQN_TARGET_UPDATE": 10,
    "DQN_Hidden": 64,
    "DQN_TAU": 0.005
}

# if cuda avail
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Geometry and Physics
class Track:
    def __init__(self, size, width):
        self.size = size
        self.width = width
        self.last_cp_time = 0

        # Define Square Track, Outer and Innner walls
        # Outer Clockwise
        self.outer_walls = [
            ((0, 0), (size, 0)),
            ((size, 0), (size, size)),
            ((size, size), (0, size)),
            ((0, size), (0, 0))
        ]
        # Inner Clockwise
        inner_s = width
        inner_e = size - width
        self.inner_walls = [
            ((inner_s, inner_s), (inner_e, inner_s)),
            ((inner_e, inner_s), (inner_e, inner_e)),
            ((inner_e, inner_e), (inner_s, inner_e)),
            ((inner_s, inner_e), (inner_s, inner_s))
        ]
        self.walls = self.outer_walls + self.inner_walls
        
        self.checkpoints = [
            (size/2, width/2),      # Bottom
            (size-width/2, size/2), # Right
            (size/2, size-width/2), # Top
            (width/2, size/2)       # Left
        ]

    def check_collision(self, x, y):
        r = 3.5 # Car radius
        # Outer bounds (100) - collision if center is within radius of 0 or 100
        if not (r <= x <= self.size - r and r <= y <= self.size - r):
            return True
        
        # Inner bounds: 20 to 80, collision if center touches expanded inner box
        inner_s, inner_e = self.width - r, (self.size - self.width) + r
        if inner_s <= x <= inner_e and inner_s <= y <= inner_e:
            return True
        
        return False

    def get_ray_intersection(self, ray_start, ray_dir):
        # Vectorized line intersection using cross product logic
        closest_dist = CONFIG["SENSOR_RANGE"]
        
        x1, y1 = ray_start
        x2, y2 = x1 + ray_dir[0] * closest_dist, y1 + ray_dir[1] * closest_dist
        
        for p1, p2 in self.walls:
            x3, y3 = p1
            x4, y4 = p2
            
            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            # Parallel lines
            if denom == 0:
                continue
            
            t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
            u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
            
            if 0 <= t <= 1 and 0 <= u <= 1:
                # Intersection found
                px = x1 + t * (x2 - x1)
                py = y1 + t * (y2 - y1)
                dist = math.sqrt((px - x1)**2 + (py - y1)**2)
                if dist < closest_dist:
                    closest_dist = dist
                    
        return closest_dist

class Car:
    def __init__(self, track):
        self.track = track
        self.reset()
        
    def reset(self):
        # Start at bottom center, facing right
        self.x = CONFIG["TRACK_SIZE"] / 2
        self.y = CONFIG["TRACK_WIDTH"] / 2
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
                self.current_checkpoint = (self.current_checkpoint + 1) % 4
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
            dist = self.track.get_ray_intersection((self.x, self.y), ray_dir)
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
    def __init__(self):
        self.population = [EvolutionNet().to(device) for _ in range(CONFIG["GA_POP_SIZE"])]
        self.gen_count = 0

        # Logger Setup
        with open("bench/ga_log.txt", "w") as f:
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
        with open("bench/ga_log.txt", "a") as f:
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
        self.fc = nn.Sequential(
            nn.Linear(inputs, CONFIG["DQN_Hidden"]),
            nn.ReLU(),
            nn.Linear(CONFIG["DQN_Hidden"], CONFIG["DQN_Hidden"]),
            nn.ReLU(),
            nn.Linear(CONFIG["DQN_Hidden"], outputs) 
        )

    def forward(self, x):
        return self.fc(x)
    
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self):
        self.policy_net = DQNNet(CONFIG["N_SENSORS"], 3).to(device)
        self.target_net = DQNNet(CONFIG["N_SENSORS"], 3).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        # Target net is never trained directly
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=CONFIG["DQN_LR"])
        self.memory = ReplayBuffer(CONFIG["DQN_MEMORY_SIZE"])
        self.steps_done = 0
        self.episode = 0
        
        # Logs
        if os.path.exists("bench/dqn_log.txt"): 
            os.remove("bench/dqn_log.txt")
        with open("bench/dqn_log.txt", "w") as f:
            f.write("Episode,Reward,Duration,Epsilon,Circles\n")

    def soft_update(self):
        target_dict = self.target_net.state_dict()
        policy_dict = self.policy_net.state_dict()
        for key in policy_dict:
            target_dict[key] = policy_dict[key] * CONFIG["DQN_TAU"] + target_dict[key] * (1 - CONFIG["DQN_TAU"])

        self.target_net.load_state_dict(target_dict)

    def select_action(self, state):
        eps = CONFIG["DQN_EPS_END"] + (CONFIG["DQN_EPS_START"] - CONFIG["DQN_EPS_END"])*math.exp(-1. * self.steps_done/ CONFIG["DQN_EPS_DECAY"])
        self.steps_done += 1
        
        if random.random() > eps:
            with torch.no_grad():
                return self.policy_net(state.unsqueeze(0)).argmax(dim=1).item()
        else:
            return random.randint(0, 2)

    def optimize_model(self):
        if len(self.memory) < CONFIG["DQN_BATCH_SIZE"]: return
        
        transitions = self.memory.sample(CONFIG["DQN_BATCH_SIZE"])
        batch = list(zip(*transitions))

        # Stack tensors
        state_batch = torch.stack(batch[0])
        action_batch = torch.LongTensor(batch[1]).unsqueeze(1).to(device)
        reward_batch = torch.FloatTensor(batch[2]).unsqueeze(1).to(device)
        next_state_batch = torch.stack(batch[3])
        done_batch = torch.FloatTensor(batch[4]).unsqueeze(1).to(device)

        '''
        Compute Q(s_t, a)the model computes Q(s_t), 
        then we select the columns of actions taken
        '''
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        '''
        Expected values of actions for non_final_next_states are computed based on the "older" target_net
        '''
        with torch.no_grad():
            next_state_values = self.target_net(next_state_batch).max(1)[0].unsqueeze(1)
            # expected Q values
            expected_state_action_values = reward_batch + (next_state_values * CONFIG["DQN_GAMMA"] * (1 - done_batch))

        # Huber loss, overcome bad gradients
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        # In place gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10)
        self.optimizer.step()
        self.soft_update()

    # def update_target_network(self):
    #     #update: Copy weights entirely
    #     self.target_net.load_state_dict(self.policy_net.state_dict())

    def log_episode(self, total_reward, duration, circles):
        eps = CONFIG["DQN_EPS_END"] + (CONFIG["DQN_EPS_START"] - CONFIG["DQN_EPS_END"])*math.exp(-1. * self.steps_done / CONFIG["DQN_EPS_DECAY"])
              
        print(f"DQN Ep {self.episode}: Rew {total_reward:.1f} | Steps {duration} | Circles {circles} | Eps {eps:.2f}")
        with open("bench/dqn_log.txt", "a") as f:
            f.write(f"{self.episode},{total_reward},{duration},{eps:.2f},{circles}\n")
        self.episode += 1

#Visualisation
def draw_track(ax, track, car, sensors=None):
    ax.clear()
    ax.set_xlim(0, track.size)
    ax.set_ylim(0, track.size)
    
    # Draw Walls
    for p1, p2 in track.outer_walls:
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', linewidth=2)
    for p1, p2 in track.inner_walls:
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', linewidth=2)
        
    # Draw Car
    circle = plt.Circle((car.x, car.y), 1.5, color='b' if car.alive else 'r')
    ax.add_patch(circle)
    
    # Draw Sensors (Viz of raycasts)
    if sensors is not None:
        start_angle = car.angle - math.pi / 2
        step_angle = math.pi / (CONFIG["N_SENSORS"] - 1)
        for i, dist_norm in enumerate(sensors):
            ray_angle = start_angle + i * step_angle
            real_dist = dist_norm * CONFIG["SENSOR_RANGE"]
            ex = car.x + math.cos(ray_angle) * real_dist
            ey = car.y + math.sin(ray_angle) * real_dist
            ax.plot([car.x, ex], [car.y, ey], 'g-', alpha=0.5)

def run_simulation():
    track = Track(CONFIG["TRACK_SIZE"], CONFIG["TRACK_WIDTH"])
    
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
                    draw_track(ax, track, demo_car, demo_car.radars)
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
            state = car.get_state()
            total_reward = 0
            
            while car.alive and car.time_alive < 10000:
                # Select action
                action = agent.select_action(state)
                
                # Repeat action for stabiltiy
                reward_accum = 0
                for _ in range(FRAME_SKIP):
                    next_state, r, done, _ = car.step(action)
                    reward_accum += r
                    if done:
                        break
                        
                # Store transitions, acc rewards
                agent.memory.push(state, action, reward_accum, next_state, done)
                state = next_state
                total_reward += reward_accum
                
                #Optimize
                agent.optimize_model()
                                
                # Vis
                if agent.episode % 10 == 0 and car.time_alive % CONFIG["RENDER_EVERY"] == 0:
                    draw_track(ax, track, car, car.radars)
                    plt.title(f"DQN Ep {agent.episode} | Rew: {total_reward:.1f}")
                    plt.pause(1/CONFIG["FPS"])
            
            agent.log_episode(total_reward, car.time_alive, car.circles)

if __name__ == "__main__":
    run_simulation()