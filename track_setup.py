import numpy as np
import math
import matplotlib.pyplot as plt


class RegularPolygonTrack:
    """
    Generates a regular n-sided polygon track with outer and inner walls.
    Also creates n checkpoints, one on each side of the polygon.
    """
    
    def __init__(self, size, width, n_sides=4):
        """
        Args:
            size: Circumradius of the polygon (distance from center to vertex)
            width: Width of the track (distance between outer and inner walls)
            n_sides: Number of sides of the polygon (default: 4 for square)
        """
        self.size = size
        self.width = width
        self.n_sides = n_sides
        self.last_cp_time = 0
        self.center = (size, size)  # Center of polygon
        
        # Generate outer and inner walls
        self.outer_walls = self._generate_polygon_walls(size, n_sides, outer=True)
        self.inner_walls = self._generate_polygon_walls(size - width, n_sides, outer=False)
        self.walls = self.outer_walls + self.inner_walls
        
        # Generate checkpoints
        self.checkpoints = self._generate_checkpoints()

    def _generate_polygon_walls(self, radius, n_sides, outer=True):
        """
        Generate wall segments for a regular n-sided polygon.
        
        Args:
            radius: Distance from center to vertex
            n_sides: Number of sides
            outer: If True, generates outer walls; if False, generates inner walls
            
        Returns:
            List of line segments (p1, p2) forming the polygon
        """
        walls = []
        center_x, center_y = self.center
        
        for i in range(n_sides):
            # Angle of current vertex
            angle1 = 2 * math.pi * i / n_sides - math.pi / 2
            # Angle of next vertex
            angle2 = 2 * math.pi * (i + 1) / n_sides - math.pi / 2
            
            # Calculate vertex positions
            x1 = center_x + radius * math.cos(angle1)
            y1 = center_y + radius * math.sin(angle1)
            x2 = center_x + radius * math.cos(angle2)
            y2 = center_y + radius * math.sin(angle2)
            
            walls.append(((x1, y1), (x2, y2)))
        
        return walls

    def _generate_checkpoints(self):
        """
        Generate n checkpoints, one at the center of each side.
        
        Returns:
            List of checkpoint coordinates
        """
        checkpoints = []
        center_x, center_y = self.center
        
        # Checkpoint is placed at the midpoint between outer and inner radius
        checkpoint_radius = (self.size + (self.size - self.width)) / 2
        
        for i in range(self.n_sides):
            # Angle pointing to the center of the side
            angle = 2 * math.pi * i / self.n_sides - math.pi / 2
            
            x = center_x + checkpoint_radius * math.cos(angle)
            y = center_y + checkpoint_radius * math.sin(angle)
            
            checkpoints.append((x, y))
        
        return checkpoints

    def check_collision(self, x, y):
        """
        Check if car collides with any polygon wall segment.
        """
        car_radius = 2.0 # Slightly larger than visual radius (1.5) for safety
        
        for p1, p2 in self.walls:
            # Vector from p1 to car
            px = x - p1[0]
            py = y - p1[1]
            # Vector from p1 to p2 (wall segment)
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            
            segment_len_sq = dx*dx + dy*dy
            
            if segment_len_sq == 0: continue # Avoid div by zero

            # Project point onto line to find 't' (parameter along the segment)
            # t must be clamped between 0 and 1 to stay within the segment endpoints
            t = max(0, min(1, (px*dx + py*dy) / segment_len_sq))
            
            # Find closest point on the segment
            closest_x = p1[0] + t * dx
            closest_y = p1[1] + t * dy
            
            # Check distance squared (faster than sqrt)
            dist_sq = (x - closest_x)**2 + (y - closest_y)**2
            
            if dist_sq < car_radius**2:
                return True
                
        return False

    def get_ray_intersection(self, ray_start, ray_dir, sensor_range):
        """
        Find the closest intersection of a ray with track walls.
        
        Args:
            ray_start: (x, y) starting position of ray
            ray_dir: (dx, dy) direction of ray
            sensor_range: Maximum range to check
            
        Returns:
            Distance to closest wall (clamped to sensor_range)
        """
        closest_dist = sensor_range
        
        x1, y1 = ray_start
        x2 = x1 + ray_dir[0] * closest_dist
        y2 = y1 + ray_dir[1] * closest_dist
        
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


def draw_track(ax, track, car, sensors=None, config=None):
    """
    Draw the track and car on the given matplotlib axes.
    
    Args:
        ax: Matplotlib axes object
        track: RegularPolygonTrack instance
        car: Car instance
        sensors: Normalized sensor readings (0-1)
        config: Configuration dict containing FPS, SENSOR_RANGE, N_SENSORS
    """
    if config is None:
        raise ValueError("config parameter is required")
    
    ax.clear()
    
    # Calculate plot limits with padding
    padding = 10
    ax.set_xlim(track.center[0] - track.size - padding, 
                track.center[0] + track.size + padding)
    ax.set_ylim(track.center[1] - track.size - padding, 
                track.center[1] + track.size + padding)
    ax.set_aspect('equal')
    
    # Draw Walls
    for p1, p2 in track.outer_walls:
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', linewidth=2)
    for p1, p2 in track.inner_walls:
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', linewidth=2)
    
    # Draw checkpoints
    for checkpoint in track.checkpoints:
        ax.plot(checkpoint[0], checkpoint[1], 'g^', markersize=8)
    
    # Draw Car
    circle = plt.Circle((car.x, car.y), 1.5, color='b' if car.alive else 'r')
    ax.add_patch(circle)
    
    # Draw car direction
    arrow_len = 3
    arrow_end_x = car.x + arrow_len * math.cos(car.angle)
    arrow_end_y = car.y + arrow_len * math.sin(car.angle)
    ax.arrow(car.x, car.y, arrow_end_x - car.x, arrow_end_y - car.y, 
             head_width=1, head_length=0.5, fc='b', ec='b')
    
    # Draw Sensors (raycasts)
    if sensors is not None:
        start_angle = car.angle - math.pi / 2
        step_angle = math.pi / (config["N_SENSORS"] - 1)
        for i, dist_norm in enumerate(sensors):
            ray_angle = start_angle + i * step_angle
            real_dist = dist_norm * config["SENSOR_RANGE"]
            ex = car.x + math.cos(ray_angle) * real_dist
            ey = car.y + math.sin(ray_angle) * real_dist
            ax.plot([car.x, ex], [car.y, ey], 'g-', alpha=0.5)


def create_track(size, width, n_sides=4):
    """
    Wrapper function to create a track with the specified number of sides.
    
    Args:
        size: Circumradius of the polygon
        width: Width of the track
        n_sides: Number of sides (default: 4 for square, can be 3, 5, 6, etc.)
        
    Returns:
        RegularPolygonTrack instance
    """
    if n_sides < 3:
        raise ValueError("n_sides must be at least 3")
    
    return RegularPolygonTrack(size, width, n_sides)
