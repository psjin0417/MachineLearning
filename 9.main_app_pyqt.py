import sys
import cv2
import numpy as np
import joblib
import pickle
import os
import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QTableWidget, 
                             QTableWidgetItem, QHeaderView, QMessageBox)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread, QSize
from skimage.feature import hog

# --- Global Constants & Parameters ---
VIDEO_PATH = "./video/10.mp4"
MODEL_PATH = "./svm_model_v3_gridsearch.pkl"
ROI_FILE = "./roi_config.pkl"
TARGET_SIZE = (128, 128)

# Detection & Visualization
RESIZE_SCALE = 0.5
ROTATE_CODE = None
SHOW_HEATMAP = False  # Default off, can be toggled
HEATMAP_THRESH = 150
CONFIDENCE_THRESHOLD = 0.98

# Pyramid
SCALE_STEP = 1.1

# HOG Params
HOG_PARAMS = {
    'orientations': 9,
    'pixels_per_cell': (16, 16),
    'cells_per_block': (2, 2),
    'block_norm': 'L2-Hys',
    'visualize': False,
    'transform_sqrt': True,
    'feature_vector': False
}
HIST_BINS = (32, 32)
PIXELS_PER_CELL = 16

# Event Logging
CONSECUTIVE_FRAME_THRESHOLD = 5  # Number of frames to trigger log

def extract_color_histogram(image, bins=(32, 32)):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, bins, [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

# --- Worker Thread for Video Processing ---
class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    detection_event_signal = pyqtSignal(str, np.ndarray) # msg, frame_snapshot

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.model = joblib.load(MODEL_PATH)
        self.roi_config = self.load_roi()
        self.consecutive_count = 0
        self.last_detection_time = datetime.datetime.min # For debouncing logs if needed

    def load_roi(self):
        if os.path.exists(ROI_FILE):
            try:
                with open(ROI_FILE, 'rb') as f:
                    return pickle.load(f)
            except:
                pass
        return None

    def set_roi(self, rois):
        self.roi_config = rois

    def run(self):
        cap = cv2.VideoCapture(VIDEO_PATH)
        
        while self._run_flag:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Loop video
                continue
            
            if ROTATE_CODE is not None:
                frame = cv2.rotate(frame, ROTATE_CODE)

            # Detect
            heatmap = self.detect_multi_scale(frame)
            
            # Post-process
            processed_frame, is_danger = self.post_process(frame, heatmap)
            
            # Event Logic
            if is_danger:
                self.consecutive_count += 1
            else:
                self.consecutive_count = 0
            
            if self.consecutive_count == CONSECUTIVE_FRAME_THRESHOLD:
                # Trigger Event (just once when threshold hit)
                timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                msg = f"Dashboard Warning ({timestamp})"
                self.detection_event_signal.emit(msg, processed_frame.copy())
            
            self.change_pixmap_signal.emit(processed_frame)
        
        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()

    def detect_multi_scale(self, frame):
        """Fast HOG Subsampling Implementation"""
        heatmap = np.zeros(frame.shape[:2], dtype=np.float32)
        img_tosearch = frame.copy()
        
        # Initial resize
        if RESIZE_SCALE != 1.0:
            width = int(img_tosearch.shape[1] * RESIZE_SCALE)
            height = int(img_tosearch.shape[0] * RESIZE_SCALE)
            img_tosearch = cv2.resize(img_tosearch, (width, height), interpolation=cv2.INTER_AREA)

        scale = 1.0
        roi_mask = None
        
        # Optimize: Create mask from 'detection' ROI if available
        if self.roi_config and 'detection' in self.roi_config:
            # Create a mask at full resolution first
            full_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.fillPoly(full_mask, [self.roi_config['detection']], 255)
            # We will accept detection if the center matches. 
            # Ideally we pass this mask down, but for speed, let's keep it simple and check later.
            roi_mask = full_mask

        while True:
            if img_tosearch.shape[0] < TARGET_SIZE[1] or img_tosearch.shape[1] < TARGET_SIZE[0]:
                break
                
            gray = cv2.cvtColor(img_tosearch, cv2.COLOR_BGR2GRAY)
            hog_features_map = hog(gray, **HOG_PARAMS)
            
            n_blocks_per_window = (TARGET_SIZE[0] // PIXELS_PER_CELL) - 1
            
            max_y_cells = hog_features_map.shape[0] - n_blocks_per_window
            max_x_cells = hog_features_map.shape[1] - n_blocks_per_window
            
            if max_y_cells < 0 or max_x_cells < 0:
                break

            for yc in range(0, max_y_cells):
                for xc in range(0, max_x_cells):
                    
                    feature_chunk = hog_features_map[yc:yc+n_blocks_per_window, 
                                                   xc:xc+n_blocks_per_window, :, :, :]
                    hog_feat = feature_chunk.ravel()
                    
                    if len(hog_feat) != 1764: continue

                    x_pixel = xc * PIXELS_PER_CELL
                    y_pixel = yc * PIXELS_PER_CELL
                    
                    window_img = img_tosearch[y_pixel : y_pixel + TARGET_SIZE[1], 
                                              x_pixel : x_pixel + TARGET_SIZE[0]]
                    
                    if window_img.shape[0] != TARGET_SIZE[1] or window_img.shape[1] != TARGET_SIZE[0]:
                        continue

                    color_feat = extract_color_histogram(window_img, bins=HIST_BINS)
                    combined = np.hstack([hog_feat, color_feat]).reshape(1, -1)
                    
                    prob = self.model.predict_proba(combined)[0][1]
                    
                    if prob > CONFIDENCE_THRESHOLD:
                        curr_scale_inv = 1.0 / (RESIZE_SCALE / scale)
                        real_x = int(x_pixel * curr_scale_inv)
                        real_y = int(y_pixel * curr_scale_inv)
                        real_w = int(TARGET_SIZE[0] * curr_scale_inv)
                        real_h = int(TARGET_SIZE[1] * curr_scale_inv)
                        
                        center_x = real_x + real_w // 2
                        center_y = real_y + real_h // 2
                        
                        # ROI Filtering
                        if roi_mask is not None:
                            # Check if center is in detection roi
                            if 0 <= center_y < roi_mask.shape[0] and 0 <= center_x < roi_mask.shape[1]:
                                if roi_mask[center_y, center_x] == 0:
                                    continue

                        real_y2 = min(real_y + real_h, frame.shape[0])
                        real_x2 = min(real_x + real_w, frame.shape[1])
                        heatmap[real_y:real_y2, real_x:real_x2] += prob

            img_tosearch = cv2.resize(img_tosearch, 
                                    (int(img_tosearch.shape[1] / SCALE_STEP), 
                                     int(img_tosearch.shape[0] / SCALE_STEP)))
            scale *= SCALE_STEP
        
        return heatmap

    def post_process(self, frame, heatmap):
        heatmap_norm = np.clip(heatmap, 0, 255)
        if np.max(heatmap_norm) > 0:
            heatmap_norm = heatmap_norm / np.max(heatmap_norm) * 255
        heatmap_norm = heatmap_norm.astype(np.uint8)
        
        _, thresh = cv2.threshold(heatmap_norm, HEATMAP_THRESH, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        result = frame.copy()
        is_danger = False

        # Draw ROI Overlay
        if self.roi_config:
            overlay = result.copy()
            if 'detection' in self.roi_config:
                cv2.fillPoly(overlay, [self.roi_config['detection']], (255, 255, 0))
                cv2.polylines(result, [self.roi_config['detection']], True, (255, 255, 0), 2)
            if 'danger' in self.roi_config:
                cv2.fillPoly(overlay, [self.roi_config['danger']], (0, 0, 255))
                cv2.polylines(result, [self.roi_config['danger']], True, (0, 0, 255), 2)
            result = cv2.addWeighted(result, 0.8, overlay, 0.2, 0)
        
        # Draw Detections
        if contours:
            max_cnt = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(max_cnt)
            cx, cy = x + w // 2, y + h // 2
            
            color = (0, 255, 0)
            status_text = "Kickboard"
            
            # Use Danger Zone logic
            if self.roi_config and 'danger' in self.roi_config:
                 if cv2.pointPolygonTest(self.roi_config['danger'], (cx, cy), False) >= 0:
                     color = (0, 0, 255)
                     status_text = "WARNING!"
                     is_danger = True
            
            cv2.rectangle(result, (x, y), (x+w, y+h), color, 2)
            cv2.putText(result, status_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        return result, is_danger

# --- Main Application Window ---
class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Smart Kickboard Detector")
        self.resize(1200, 700)
        
        # Layouts
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Left: Video Section
        video_layout = QVBoxLayout()
        self.image_label = QLabel("Click 'Start Video' to begin")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: #222; color: #eee; font-size: 16px;")
        self.image_label.setFixedSize(800, 600)
        video_layout.addWidget(self.image_label)
        
        # Controls
        controls_layout = QHBoxLayout()
        
        self.btn_start = QPushButton("Start Video")
        self.btn_start.clicked.connect(self.start_video)
        
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.clicked.connect(self.stop_video)
        self.btn_stop.setEnabled(False)
        
        self.btn_roi = QPushButton("Set ROI")
        self.btn_roi.clicked.connect(self.set_roi_mode)
        
        controls_layout.addWidget(self.btn_start)
        controls_layout.addWidget(self.btn_stop)
        controls_layout.addWidget(self.btn_roi)
        video_layout.addLayout(controls_layout)
        
        main_layout.addLayout(video_layout, stretch=2)
        
        # Right: Log Section
        log_layout = QVBoxLayout()
        log_label = QLabel("Event Log")
        log_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        log_layout.addWidget(log_label)
        
        self.log_table = QTableWidget()
        self.log_table.setColumnCount(2)
        self.log_table.setHorizontalHeaderLabels(["Time", "Snapshot"])
        self.log_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.log_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.log_table.setIconSize(QSize(100, 80)) # Larger icon for snapshot
        self.log_table.verticalHeader().setDefaultSectionSize(80) 
        
        log_layout.addWidget(self.log_table)
        main_layout.addLayout(log_layout, stretch=1)
        
        # Thread
        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.detection_event_signal.connect(self.add_log_entry)

    def start_video(self):
        if not self.thread.isRunning():
            self.thread._run_flag = True
            self.thread.start()
            self.btn_start.setEnabled(False)
            self.btn_stop.setEnabled(True)
            self.btn_roi.setEnabled(False) # ROI setting blocked while running

    def stop_video(self):
        if self.thread.isRunning():
            self.thread.stop()
            self.btn_start.setEnabled(True)
            self.btn_stop.setEnabled(False)
            self.btn_roi.setEnabled(True)

    def closeEvent(self, event):
        self.stop_video()
        event.accept()

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(800, 600, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)
    
    @pyqtSlot(str, np.ndarray)
    def add_log_entry(self, msg, frame):
        row = self.log_table.rowCount()
        self.log_table.insertRow(row)
        
        # Time Item
        time_item = QTableWidgetItem(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        time_item.setTextAlignment(Qt.AlignCenter)
        self.log_table.setItem(row, 0, time_item)
        
        # Snapshot Item (Thumbnail)
        # Crop center or resize for icon
        h, w, _ = frame.shape
        thumb = cv2.resize(frame, (100, 75))
        thumb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
        img = QImage(thumb.data, 100, 75, 3*100, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(img)
        
        icon_item = QTableWidgetItem()
        # Fix: need QtGui.QIcon
        from PyQt5.QtGui import QIcon
        icon_item.setIcon(QIcon(pixmap))
        icon_item.setText("Warning Detected")
        self.log_table.setItem(row, 1, icon_item)
        
        self.log_table.scrollToBottom()

    def set_roi_mode(self):
        # Stop thread if running (safety)
        if self.thread.isRunning():
            self.thread.stop()
        
        # Open OpenCV window for ROI
        cap = cv2.VideoCapture(VIDEO_PATH)
        ret, frame = cap.read()
        cap.release()
        
        if not ret: return
        if ROTATE_CODE: frame = cv2.rotate(frame, ROTATE_CODE)

        # Reuse drawing logic (simplified here)
        msg_box = QMessageBox()
        msg_box.setText("OpenCV 창에서 ROI를 설정합니다.\n1. Detection Zone 설정\n2. Danger Zone 설정")
        msg_box.exec_()
        
        det_roi = self._get_poly_from_cv(frame, "Set Detection ROI (Green)")
        if det_roi is None: return
        
        danger_roi = self._get_poly_from_cv(frame, "Set Danger Zone (Red)")
        if danger_roi is None: return
        
        new_rois = {'detection': det_roi, 'danger': danger_roi}
        
        # Save
        with open(ROI_FILE, 'wb') as f:
            pickle.dump(new_rois, f)
        
        # Update Thread
        self.thread.set_roi(new_rois)
        QMessageBox.information(self, "Success", "ROI 설정이 저장되었습니다.")

    def _get_poly_from_cv(self, frame, title):
        points = []
        def click_event(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append((x, y))

        window_name = title
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, click_event)
        
        while True:
            disp = frame.copy()
            if len(points) > 0:
                pts = np.array(points, np.int32).reshape((-1, 1, 2))
                cv2.polylines(disp, [pts], True, (0, 255, 0), 2)
                for pt in points: cv2.circle(disp, pt, 5, (0, 0, 255), -1)
            
            cv2.putText(disp, "Click points. Press 'n' to finish.", (30, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.imshow(window_name, disp)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('n'): break
            
        cv2.destroyWindow(window_name)
        return np.array(points, dtype=np.int32) if len(points) >= 3 else None

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = App()
    window.show()
    sys.exit(app.exec_())
