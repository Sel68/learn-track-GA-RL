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