# Rover LIDAR Mapping System Configuration

# D500 LIDAR Settings
LIDAR_PORT = 'COM3'              # Serial port for LIDAR (check Device Manager)
LIDAR_BAUDRATE = 230400          # D500 default baudrate
LIDAR_MAX_RANGE = 8.0            # Maximum range in meters (from specs)

# Camera Settings
CAMERA_ID = 1                    # Camera device ID (0 for default)
CAMERA_WIDTH = 1280              # Camera resolution width (720p)
CAMERA_HEIGHT = 720              # Camera resolution height (720p)

# Mapping Parameters
MAP_SIZE = 800                   # Grid size in pixels (800x800)
MAP_RESOLUTION = 0.02            # Meters per pixel (2cm resolution)
MAP_UPDATE_RATE = 60             # Hz (updates per second)

# Visualization
WINDOW_WIDTH = 1600
WINDOW_HEIGHT = 900
SHOW_FPS = True
SHOW_CAMERA = True

# Performance
USE_THREADING = True             # Enable parallel processing
MAX_SCAN_BUFFER = 100            # Number of scans to buffer

# Path Planning (future use)
SAFETY_DISTANCE = 0.3            # Minimum clearance in meters
PATH_SMOOTHING = True
