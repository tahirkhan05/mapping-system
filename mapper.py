"""
Efficient real-time SLAM-lite mapper for rover navigation
Uses occupancy grid with fast updates and obstacle classification
"""
import numpy as np
import cv2
from collections import deque


class RoverMapper:
    def __init__(self, map_size=800, resolution=0.02, max_range=8.0):
        """
        Args:
            map_size: Grid size in pixels (800x800)
            resolution: Meters per pixel (0.02m = 2cm)
            max_range: Maximum LIDAR range in meters (8m from specs)
        """
        self.map_size = map_size
        self.resolution = resolution
        self.max_range = max_range
        self.center = map_size // 2
        
        # Occupancy grid: 0=unknown, 128=free, 255=occupied
        self.grid = np.ones((map_size, map_size), dtype=np.uint8) * 0  # Start with unknown
        
        # Decay factor for temporal filtering
        self.decay_rate = 0.95  # Fade old data
        
        # Boundary detection
        self.boundaries = set()
        self.obstacles = []
        
        # Path history for rover position
        self.path_history = deque(maxlen=500)
        self.rover_pos = (self.center, self.center)
        self.rover_angle = 0
        
    def world_to_grid(self, x, y):
        """Convert world coordinates (meters) to grid coordinates (pixels)"""
        grid_x = int(self.center + x / self.resolution)
        grid_y = int(self.center - y / self.resolution)  # Flip Y axis
        return grid_x, grid_y
    
    def update_map(self, lidar_x, lidar_y, rover_x=0, rover_y=0, rover_theta=0):
        """Update occupancy grid with new LIDAR scan"""
        if len(lidar_x) == 0:
            return
        
        # Aggressive decay to remove stale data when moving
        self.grid = (self.grid * 0.85).astype(np.uint8)  # Faster fade
        
        # Update rover position
        new_rover_pos = self.world_to_grid(rover_x, rover_y)
        
        # Clear area around old rover position to prevent artifacts
        if hasattr(self, 'rover_pos'):
            old_x, old_y = self.rover_pos
            clear_radius = 20
            for dx in range(-clear_radius, clear_radius):
                for dy in range(-clear_radius, clear_radius):
                    cx, cy = old_x + dx, old_y + dy
                    if 0 <= cx < self.map_size and 0 <= cy < self.map_size:
                        if dx*dx + dy*dy <= clear_radius*clear_radius:
                            self.grid[cy, cx] = np.uint8(self.grid[cy, cx] * 0.5)
        
        self.rover_pos = new_rover_pos
        self.rover_angle = rover_theta
        self.path_history.append(self.rover_pos)
        
        # Clear obstacles list and boundaries for current scan
        self.obstacles = []
        self.boundaries = set()
        
        # Process each LIDAR point
        for x, y in zip(lidar_x, lidar_y):
            # Transform to world frame
            wx = x + rover_x
            wy = y + rover_y
            
            gx, gy = self.world_to_grid(wx, wy)
            
            # Check bounds
            if 0 <= gx < self.map_size and 0 <= gy < self.map_size:
                # Mark obstacle
                distance = np.sqrt(x**2 + y**2)
                if distance < self.max_range and distance > 0.05:  # Valid range
                    self.grid[gy, gx] = 255  # Occupied
                    self.obstacles.append((gx, gy))
                    
                    # Detect boundaries (strong reflections at edges)
                    if distance > 0.3:  # Ignore very close points
                        self.boundaries.add((gx, gy))
                    
                    # Only trace rays for reliable measurements
                    if distance > 0.1:  # Skip very close noisy readings
                        self._bresenham_line(self.rover_pos[0], self.rover_pos[1], gx, gy)
    
    def _bresenham_line(self, x0, y0, x1, y1):
        """Fast line drawing for ray tracing (mark free space)"""
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        step_count = 0
        max_steps = max(dx, dy)
        
        while True:
            # Mark as free (but don't overwrite fresh obstacles)
            if 0 <= x0 < self.map_size and 0 <= y0 < self.map_size:
                current_val = int(self.grid[y0, x0])
                if current_val < 250:  # Don't overwrite current obstacles
                    # Lighter marking for free space, easier to clear
                    self.grid[y0, x0] = np.uint8(min(150, current_val + 15))
            
            if x0 == x1 and y0 == y1:
                break
            
            step_count += 1
            if step_count > max_steps:  # Safety limit
                break
                
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
    
    def get_visualization(self):
        """Generate color-coded map visualization (like reference image)"""
        # Create RGB image
        vis = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)
        
        # Color scheme: Blue=free, Red=obstacles/walls, Black=unknown
        # Free space (gradient based on confidence)
        free_mask = (self.grid > 50) & (self.grid < 250)
        vis[:, :, 0] = np.where(free_mask, np.clip(self.grid * 1.7, 0, 255), 0)  # Blue channel
        
        # Obstacles (bright red)
        obstacle_mask = self.grid >= 250
        vis[:, :, 2] = np.where(obstacle_mask, 255, 0)  # Red channel
        
        # Highlight boundaries in bright red
        for bx, by in list(self.boundaries)[-1000:]:  # Limit for performance
            if 0 <= bx < self.map_size and 0 <= by < self.map_size:
                cv2.circle(vis, (bx, by), 2, (0, 0, 255), -1)
        
        # Draw rover position (yellow circle)
        cv2.circle(vis, self.rover_pos, 8, (0, 255, 255), -1)
        
        # Draw rover direction indicator
        end_x = int(self.rover_pos[0] + 15 * np.cos(self.rover_angle))
        end_y = int(self.rover_pos[1] - 15 * np.sin(self.rover_angle))
        cv2.line(vis, self.rover_pos, (end_x, end_y), (0, 255, 255), 2)
        
        # Draw path history (green trail)
        if len(self.path_history) > 1:
            points = np.array(list(self.path_history), dtype=np.int32)
            cv2.polylines(vis, [points], False, (0, 255, 0), 1)
        
        return vis
    
    def get_obstacle_clusters(self):
        """Group nearby obstacles for object detection"""
        if not self.obstacles:
            return []
        
        obstacles = np.array(self.obstacles)
        # Simple clustering by proximity
        clusters = []
        used = set()
        
        for i, (x, y) in enumerate(obstacles):
            if i in used:
                continue
            
            cluster = [(x, y)]
            used.add(i)
            
            # Find nearby points
            for j, (ox, oy) in enumerate(obstacles):
                if j not in used:
                    dist = np.sqrt((x - ox)**2 + (y - oy)**2)
                    if dist < 10:  # Within 10 pixels
                        cluster.append((ox, oy))
                        used.add(j)
            
            if len(cluster) > 5:  # Valid object
                clusters.append(cluster)
        
        return clusters
    
    def is_path_clear(self, target_angle, distance=1.0):
        """Check if path in given direction is clear"""
        steps = int(distance / self.resolution)
        
        for i in range(1, steps):
            check_x = int(self.rover_pos[0] + i * np.cos(target_angle))
            check_y = int(self.rover_pos[1] - i * np.sin(target_angle))
            
            if 0 <= check_x < self.map_size and 0 <= check_y < self.map_size:
                if self.grid[check_y, check_x] == 255:
                    return False
        return True
