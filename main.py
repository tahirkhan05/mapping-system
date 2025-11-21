"""
Main Rover Mapping System
Integrates LIDAR, camera, and real-time visualization with minimal latency
"""
import time
import numpy as np
from lidar_driver import D500Lidar
from mapper import RoverMapper
from visualizer import Visualizer, CameraProcessor
import config


class RoverMappingSystem:
    def __init__(self, lidar_port='COM3', camera_id=0):
        """Initialize all components"""
        print("Initializing Rover Mapping System...")
        
        # Core components
        self.lidar = D500Lidar(port=lidar_port)
        self.mapper = RoverMapper(map_size=800, resolution=0.02, max_range=8.0)
        self.visualizer = Visualizer()
        self.camera = CameraProcessor(camera_id=camera_id)
        
        # System state
        self.running = False
        self.scan_count = 0
        self.start_time = None
        
    def start(self):
        """Start all system components"""
        print("\n=== Starting System Components ===")
        
        # Start LIDAR
        print("Connecting to D500 LIDAR...")
        if not self.lidar.start():
            print("ERROR: Failed to start LIDAR!")
            return False
        print("✓ LIDAR connected")
        
        # Start camera (non-critical)
        print("Initializing camera...")
        if self.camera.start():
            print("✓ Camera ready")
        else:
            print("⚠ Camera not available (continuing without it)")
        
        # Start visualization
        print("Starting visualization...")
        self.visualizer.start()
        print("✓ Visualization ready")
        
        self.running = True
        self.start_time = time.time()
        print("\n=== System Running ===\n")
        return True
    
    def run(self):
        """Main processing loop - optimized for minimal latency"""
        try:
            while self.running:
                loop_start = time.time()
                
                # Get LIDAR data
                x_points, y_points = self.lidar.get_cartesian_points()
                
                if len(x_points) > 0:
                    self.scan_count += 1
                    
                    # Update map (fast occupancy grid update)
                    self.mapper.update_map(x_points, y_points)
                    
                    # Generate visualization
                    map_vis = self.mapper.get_visualization()
                    self.visualizer.update_map(map_vis)
                    
                    # Process camera frame
                    camera_frame = self.camera.get_frame()
                    if camera_frame is not None:
                        obstacles = self.camera.detect_obstacles(camera_frame)
                        annotated = self.camera.annotate_frame(camera_frame, obstacles)
                        self.visualizer.update_camera(annotated)
                    
                    # Update info panel
                    obstacle_clusters = self.mapper.get_obstacle_clusters()
                    runtime = time.time() - self.start_time
                    
                    info = {
                        "Runtime": f"{runtime:.1f}s",
                        "Scans": self.scan_count,
                        "LIDAR Points": len(x_points),
                        "Obstacles": len(obstacle_clusters),
                        "Boundaries": len(self.mapper.boundaries),
                    }
                    self.visualizer.set_info(info)
                
                # Render display
                key = self.visualizer.render()
                
                # Handle keyboard input
                if key == ord('q') or key == 27:  # Q or ESC
                    print("\nShutdown requested...")
                    break
                elif key == ord('r'):  # Reset map
                    self.mapper = RoverMapper(map_size=800, resolution=0.02, max_range=8.0)
                    print("Map reset")
                elif key == ord('s'):  # Save map
                    filename = f"map_{int(time.time())}.png"
                    map_vis = self.mapper.get_visualization()
                    import cv2
                    cv2.imwrite(filename, map_vis)
                    print(f"Map saved: {filename}")
                
                # Maintain target loop rate (60 FPS for smooth visualization)
                loop_time = time.time() - loop_start
                target_time = 1.0 / 60.0  # 60 Hz target
                if loop_time < target_time:
                    time.sleep(target_time - loop_time)
                    
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.stop()
    
    def stop(self):
        """Cleanup all resources"""
        print("\n=== Shutting Down ===")
        self.running = False
        
        print("Stopping LIDAR...")
        self.lidar.stop()
        
        print("Stopping camera...")
        self.camera.stop()
        
        print("Closing visualization...")
        self.visualizer.stop()
        
        print("✓ Shutdown complete")


def main():
    """Entry point"""
    print("=" * 50)
    print("  ROVER LIDAR MAPPING SYSTEM")
    print("  D500 LIDAR + Camera Integration")
    print("=" * 50)
    print("\nControls:")
    print("  Q or ESC - Quit")
    print("  R - Reset map")
    print("  S - Save current map")
    print("=" * 50)
    
    # Configuration from config.py
    LIDAR_PORT = config.LIDAR_PORT
    CAMERA_ID = config.CAMERA_ID
    
    print(f"\nUsing LIDAR Port: {LIDAR_PORT}")
    print(f"Using Camera ID: {CAMERA_ID}")
    print()
    
    # Create and run system
    system = RoverMappingSystem(lidar_port=LIDAR_PORT, camera_id=CAMERA_ID)
    
    if system.start():
        system.run()
    else:
        print("Failed to start system")


if __name__ == "__main__":
    main()
