import serial
import struct
import numpy as np
from threading import Thread, Lock
from collections import deque


class D500Lidar:
    HEADER = 0x54
    
    def __init__(self, port='COM3', baudrate=230400):
        self.port = port
        self.baudrate = baudrate
        self.serial = None
        self.running = False
        self.scan_data = deque(maxlen=100)  # Store last 100 scans
        self.lock = Lock()
        self.current_scan = []
        
    def connect(self, retries=3):
        """Establish serial connection with retry logic"""
        import time
        for attempt in range(retries):
            try:
                if self.serial and self.serial.is_open:
                    self.serial.close()
                    time.sleep(0.5)
                
                self.serial = serial.Serial(
                    self.port, 
                    self.baudrate, 
                    timeout=0.01,
                    bytesize=serial.EIGHTBITS,
                    parity=serial.PARITY_NONE,
                    stopbits=serial.STOPBITS_ONE
                )
                print(f"✓ Connected to {self.port}")
                return True
            except Exception as e:
                if attempt < retries - 1:
                    print(f"Attempt {attempt + 1} failed, retrying...")
                    time.sleep(1)
                else:
                    print(f"Connection failed after {retries} attempts: {e}")
                    return False
    
    def _parse_packet(self, data):
        """Parse D500 LIDAR packet format"""
        if len(data) < 47 or data[0] != self.HEADER:
            return None
        
        points = []
        speed = struct.unpack('<H', data[2:4])[0]  # RPM
        start_angle = struct.unpack('<H', data[4:6])[0] / 100.0
        
        # Parse 12 measurement points per packet
        for i in range(12):
            offset = 6 + i * 3
            distance = struct.unpack('<H', data[offset:offset+2])[0]  # mm
            intensity = data[offset+2]
            
            if distance > 0:  # Valid measurement
                angle = (start_angle + i * 0.72) % 360  # 0.72° resolution
                points.append({
                    'angle': np.radians(angle),
                    'distance': distance / 1000.0,  # Convert to meters
                    'intensity': intensity
                })
        
        return points
    
    def _read_loop(self):
        """Continuous read loop for LIDAR data"""
        buffer = bytearray()
        
        while self.running:
            try:
                if self.serial.in_waiting:
                    chunk = self.serial.read(self.serial.in_waiting)
                    buffer.extend(chunk)
                    
                    # Process complete packets (47 bytes each)
                    while len(buffer) >= 47:
                        # Find packet start
                        if buffer[0] == self.HEADER:
                            packet = buffer[:47]
                            buffer = buffer[47:]
                            
                            points = self._parse_packet(packet)
                            if points:
                                self.current_scan.extend(points)
                                
                                # Complete scan when we've made full rotation
                                if len(self.current_scan) > 400:  # ~360°
                                    with self.lock:
                                        self.scan_data.append(self.current_scan.copy())
                                    self.current_scan = []
                        else:
                            buffer = buffer[1:]  # Skip invalid byte
                            
            except Exception as e:
                print(f"Read error: {e}")
                
    def start(self):
        """Start LIDAR data acquisition"""
        if not self.serial or not self.serial.is_open:
            if not self.connect():
                return False
        
        self.running = True
        self.thread = Thread(target=self._read_loop, daemon=True)
        self.thread.start()
        return True
    
    def stop(self):
        """Stop LIDAR acquisition"""
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=1.0)
        if self.serial and self.serial.is_open:
            self.serial.close()
    
    def get_latest_scan(self):
        """Get most recent complete scan"""
        with self.lock:
            if self.scan_data:
                return self.scan_data[-1]
        return []
    
    def get_cartesian_points(self):
        """Convert polar to cartesian coordinates"""
        scan = self.get_latest_scan()
        if not scan:
            return np.array([]), np.array([])
        
        x = np.array([p['distance'] * np.cos(p['angle']) for p in scan])
        y = np.array([p['distance'] * np.sin(p['angle']) for p in scan])
        return x, y
