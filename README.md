# Rover LIDAR Mapping System

Real-time LIDAR mapping system for autonomous rover navigation using LDROBOT D500 LIDAR and camera.

## Hardware Requirements
- **LDROBOT D500 LIDAR Kit**
  - Detection range: 0.05m - 8m
  - Scan frequency: 6-13 Hz
  - Ranging frequency: 5000 Hz
  - Angular resolution: 0.72°
  - Interface: Serial/UART (230400 baud)
  
- **Camera** (USB/CSI compatible)
- **Serial connection** (USB-to-Serial or direct UART)

## Features
- ✅ Real-time LIDAR data acquisition (5000 Hz ranging)
- ✅ Occupancy grid mapping with obstacle detection
- ✅ Boundary/wall detection and tracking
- ✅ Camera integration for visual confirmation
- ✅ Live visualization (similar to reference image)
- ✅ Low-latency processing (~30 FPS)
- ✅ Minimal code footprint (~500 lines total)

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Configure your hardware:
   - Edit `config.py` to set correct COM port for LIDAR
   - Adjust camera ID if needed (default is 0)

## Usage

### Quick Start
```bash
python main.py
```

### Controls
- **Q** or **ESC** - Quit application
- **R** - Reset map
- **S** - Save current map as PNG

### Configuration
Edit `config.py` to customize:
- LIDAR serial port (check Windows Device Manager)
- Camera device ID
- Map size and resolution
- Update rates

## Architecture

```
main.py           - Main application loop and integration
├── lidar_driver.py   - D500 LIDAR serial communication
├── mapper.py         - Occupancy grid mapping and SLAM
├── visualizer.py     - Real-time display and camera processing
└── config.py         - Configuration parameters
```

### Performance Optimizations
- **Threaded I/O**: Parallel LIDAR reading and camera capture
- **Fast ray tracing**: Bresenham algorithm for free space marking
- **Efficient data structures**: NumPy arrays and deques
- **Minimal copying**: Direct buffer updates
- **Target 30 Hz**: Balanced visualization and processing

## Troubleshooting

### LIDAR not connecting
1. Check COM port in Device Manager
2. Verify D500 is powered (5V input)
3. Update `LIDAR_PORT` in `config.py`
4. Test with: `python -m serial.tools.list_ports`

### Camera not working
- System will continue without camera
- Check camera ID: try 0, 1, or 2
- Verify camera permissions

### High latency
- Reduce `MAP_SIZE` in config.py (e.g., 600)
- Lower camera resolution
- Disable camera: set `SHOW_CAMERA = False`

## Data Format

### LIDAR Output
Each scan provides:
- Angle: 0-360° (in radians)
- Distance: 0.05-8.0m (meters)
- Intensity: 0-255 (reflection strength)

### Map Representation
- Blue: Free navigable space
- Red: Obstacles and walls
- Yellow: Rover position and heading
- Green: Path history trail
- Black: Unknown/unexplored areas

## Future Enhancements
- Path planning algorithms
- Autonomous navigation
- Object classification
- SLAM loop closure
- Map persistence/loading

## License
MIT License - Free for educational and commercial use

## Author
Rover Mapping System v3
Built for LDROBOT D500 Integration
