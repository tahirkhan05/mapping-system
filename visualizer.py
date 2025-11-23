import cv2
import numpy as np
from threading import Thread, Lock
import time
from ultralytics import YOLO


class Visualizer:
    def __init__(self, window_name="Rover LIDAR Map"):
        self.window_name = window_name
        self.running = False
        self.lock = Lock()
        
        # Display components
        self.map_image = None
        self.camera_image = None
        self.info_text = []
        
        # Performance metrics
        self.fps = 0
        self.last_update = time.time()
        
    def start(self):
        """Start visualization window"""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1600, 900)
        self.running = True
        
    def update_map(self, map_img):
        """Update map display"""
        with self.lock:
            self.map_image = map_img.copy()
            
    def update_camera(self, cam_img):
        """Update camera feed"""
        with self.lock:
            # Resize camera to larger high-quality size
            h, w = cam_img.shape[:2]
            scale = 720 / w  # Much larger camera view
            new_size = (720, int(h * scale))
            self.camera_image = cv2.resize(cam_img, new_size, interpolation=cv2.INTER_LINEAR)
    
    def set_info(self, info_dict):
        """Set info text to display"""
        self.info_text = [f"{k}: {v}" for k, v in info_dict.items()]
    
    def render(self):
        """Render complete display frame"""
        with self.lock:
            if self.map_image is None:
                return
            
            # Create main display canvas (larger resolution)
            canvas = np.zeros((900, 1600, 3), dtype=np.uint8)
            
            # Place map (left side, square)
            map_h, map_w = self.map_image.shape[:2]
            canvas[:map_h, :map_w] = self.map_image
            
            # Add camera feed (top right, larger, moved down slightly)
            if self.camera_image is not None:
                cam_h, cam_w = self.camera_image.shape[:2]
                cam_y_offset = 30  # Move camera down
                canvas[cam_y_offset:cam_y_offset+cam_h, 850:850+cam_w] = self.camera_image
            
            # Add stylish info panel (bottom right)
            y_offset = cam_y_offset + cam_h + 40 if self.camera_image is not None else 50
            panel_height = 900 - y_offset - 20
            cv2.rectangle(canvas, (850, y_offset), (1580, 880), (30, 30, 30), -1)
            cv2.rectangle(canvas, (850, y_offset), (1580, 880), (0, 255, 255), 2)
            
            # Title with accent
            cv2.putText(canvas, "SYSTEM STATUS", (870, y_offset + 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.line(canvas, (870, y_offset + 50), (1550, y_offset + 50), (0, 255, 255), 2)
            
            # FPS with color coding
            fps_color = (0, 255, 0) if self.fps > 50 else (0, 255, 255) if self.fps > 30 else (0, 165, 255)
            cv2.putText(canvas, f"FPS: {self.fps:.1f}", (870, y_offset + 85),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.75, fps_color, 2, cv2.LINE_AA)
            
            # System info (left column)
            for i, text in enumerate(self.info_text):
                y = y_offset + 125 + i * 35
                cv2.putText(canvas, text, (870, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Add controls guide (right column, aligned with metrics)
            controls_x = 1210  # Right column position
            cv2.putText(canvas, "CONTROLS:", (controls_x, y_offset + 85),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2, cv2.LINE_AA)
            
            controls = [
                "Q / ESC - Quit",
                "R - Reset Map",
                "S - Save Map"
            ]
            for i, ctrl in enumerate(controls):
                cv2.putText(canvas, ctrl, (controls_x, y_offset + 125 + i * 35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Add legend inside system panel (below both columns)
            legend_y = y_offset + 125 + max(len(self.info_text), len(controls)) * 35 + 25
            cv2.putText(canvas, "MAP LEGEND:", (870, legend_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2, cv2.LINE_AA)
            self._draw_legend(canvas, 870, legend_y + 20)
            
            # Calculate FPS
            current_time = time.time()
            self.fps = 1.0 / (current_time - self.last_update + 0.001)
            self.last_update = current_time
            
            # Display
            cv2.imshow(self.window_name, canvas)
            key = cv2.waitKey(1) & 0xFF
            
            return key
    
    def _draw_legend(self, canvas, x, y):
        """Draw map legend in 2 columns"""
        items = [
            ("Free Space", (255, 0, 0)),      # Blue
            ("Obstacles/Walls", (0, 0, 255)),  # Red
            ("Rover Position", (0, 255, 255)), # Yellow
        ]
        
        for i, (label, color) in enumerate(items):
            # Arrange in 2 columns
            col = i % 2
            row = i // 2
            x_pos = x + col * 340
            y_pos = y + row * 32
            
            cv2.rectangle(canvas, (x_pos, y_pos), (x_pos + 25, y_pos + 15), color, -1)
            cv2.rectangle(canvas, (x_pos, y_pos), (x_pos + 25, y_pos + 15), (100, 100, 100), 2)
            cv2.putText(canvas, label, (x_pos + 35, y_pos + 12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.58, (240, 240, 240), 2, cv2.LINE_AA)
    
    def add_overlay_text(self, canvas, text, position, color=(0, 255, 0)):
        """Add overlay text to visualization"""
        cv2.putText(canvas, text, position,
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    def stop(self):
        """Close visualization"""
        self.running = False
        cv2.destroyAllWindows()


class CameraProcessor:
    def __init__(self, camera_id=0, width=1280, height=720):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.cap = None
        self.running = False
        self.latest_frame = None
        self.lock = Lock()
        
        # Initialize YOLO11s model
        print("Loading YOLO11s model...")
        self.yolo_model = YOLO('yolo11s.pt')
        self.yolo_model.fuse()
        print("✓ YOLO11s loaded")
        
    def start(self):
        """Start camera capture"""
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, 60)
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
            
            if not self.cap.isOpened():
                print("Warning: Camera not available")
                return False
            
            self.running = True
            self.thread = Thread(target=self._capture_loop, daemon=True)
            self.thread.start()
            return True
        except Exception as e:
            print(f"Camera error: {e}")
            return False
    
    def _capture_loop(self):
        """Continuous camera capture"""
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.latest_frame = frame
            time.sleep(0.016)  # ~60 FPS
    
    def get_frame(self):
        """Get latest camera frame"""
        with self.lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy()
        return None
    
    def detect_obstacles(self, frame):
        """YOLO11s-based object detection"""
        if frame is None:
            return []
        
        # Run YOLO inference
        results = self.yolo_model(frame, verbose=False, conf=0.3)[0]
        
        obstacles = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = results.names[cls]
            
            obstacles.append({
                'bbox': (x1, y1, x2 - x1, y2 - y1),
                'confidence': conf,
                'class': label
            })
        
        return obstacles
    
    def annotate_frame(self, frame, obstacles):
        """Draw YOLO detections with labels and confidence"""
        annotated = frame.copy()
        
        for obj in obstacles:
            x, y, w, h = obj['bbox']
            conf = obj['confidence']
            label = obj['class']
            
            # Color coding by object type
            if label == 'person':
                color = (0, 255, 255)  # Yellow for people
            elif label in ['car', 'truck', 'bus']:
                color = (0, 0, 255)  # Red for vehicles
            else:
                color = (0, 255, 0)  # Green for other objects
            
            # Draw thicker bounding box
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 3)
            
            # Draw larger label with background
            text = f"{label} {conf:.2f}"
            font_scale = 0.9
            thickness = 2
            (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            
            # Draw background rectangle with padding
            padding = 8
            cv2.rectangle(annotated, (x, y - text_h - padding * 2), 
                         (x + text_w + padding * 2, y), color, -1)
            cv2.rectangle(annotated, (x, y - text_h - padding * 2), 
                         (x + text_w + padding * 2, y), (255, 255, 255), 2)
            
            # Draw text in black
            cv2.putText(annotated, text, (x + padding, y - padding),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
        
        return annotated
    
    def stop(self):
        """Stop camera capture"""
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
