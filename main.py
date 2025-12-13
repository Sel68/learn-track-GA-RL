import os
import torch.nn as nn

CONFIG = {

}

class EvolutionNet(nn.Module):
    def __init__(self):
        super(EvolutionNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(CONFIG["N_SENSORS"], 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 2), # Steering, Accel
            nn.Tanh() # contraint between -1 and 1
        )
        
    def forward(self, x):
        return self.fc(x)
    
    #Geometry and Physics
class Track:
    def __init__(self, size, width):
        self.size = size
        self.width = width
        # Define Square Track, Outer and Innner walls
        # Outer Clockwise
        self.outer_walls = [
            ((0, 0), (size, 0)),
            ((size, 0), (size, size)),
            ((size, size), (0, size)),
            ((0, size), (0, 0))
        ]
        # Inner Clockwise (inset by width)
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
        r = 3 # Car radius
        # Outer bounds (100) - collision if center is within radius of 0 or 100
        if not (r <= x <= self.size - r and r <= y <= self.size - r):
            return True
        
        # Inner bounds (20 to 80) - collision if center touches expanded inner box
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
        """
        Action handling:
        GA: action is [steering, acceleration] (continuous)
        DQN: action is int (0=Left, 1=Straight, 2=Right)
        """
        if not self.alive:
            return self.get_state(), 0, True, {}

        #Physics Update (ass: bicycle model)
        steering = 0
        # GA
        if isinstance(action, (list, np.ndarray, torch.Tensor)):
            steering = float(action[0]) * 0.1 # Scale -1 to 1 -> small angle
            # (Ignoring acceleration for simpler stable training, fixed speed)
            # accel = action[1] 
        else: # DQN (Discrete)
            if action == 0: steering = -0.1
            elif action == 2: steering = 0.1
        
        self.angle += steering
        self.x += math.cos(self.angle) * self.speed
        self.y += math.sin(self.angle) * self.speed
        
        self.time_alive += 1
        self.distance_traveled += self.speed

        # Collision Detection
        if self.track.check_collision(self.x, self.y):
            self.alive = False
            reward = -10
        else:
            # Survival reward
            reward = 1 
            # Checkpoint logic for circles
            cx, cy = self.track.checkpoints[self.current_checkpoint]
            dist_to_cp = math.sqrt((self.x - cx)**2 + (self.y - cy)**2)
            if dist_to_cp < CONFIG["TRACK_WIDTH"]:
                self.current_checkpoint = (self.current_checkpoint + 1) % 4
                reward += 10 # Bonus for progress
                if self.current_checkpoint == 0:
                    self.circles += 1
                    reward += 50 # Big bonus for lap
        
        # Update Sensors
        self.radars = self._sense()
        
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