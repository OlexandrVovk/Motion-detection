#!/usr/bin/env python3
"""
Multi-Scale Pyramid Motion Detection with Track Coasting

This script performs motion detection using:
- Single optical flow at upscaled resolution (1.5x)
- Multi-scale pyramid detection (1.5x -> 1.0x -> 0.5x)
- EMA temporal accumulation
- Track Coasting with Kalman filtering
- Multi-hypothesis detection at multiple sigma thresholds
- Hungarian algorithm for detection-to-track association

Supports both video files and image sequence folders.

Usage:
    # Process video file
    python motion_detection.py --input data/videos/cars_1.mp4

    # Process image folder
    python motion_detection.py --input data/videos/I_BS_01/I_BS_01

    # With options
    python motion_detection.py --input data/videos/cars_1.mp4 --debug
    python motion_detection.py --input data/videos/I_BS_01/I_BS_01 --no-tracking
    python motion_detection.py --help

Author: Motion Detection Pipeline
"""

import argparse
import csv
import cv2
import numpy as np
from pathlib import Path
import re
from collections import deque
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Union
from scipy.optimize import linear_sum_assignment


# ============================================
# Image Sequence Reader
# ============================================

def natural_sort_key(s: str) -> List:
    """
    Key function for natural sorting of strings containing numbers.
    E.g., sorts ['img-1', 'img-2', 'img-10'] correctly instead of ['img-1', 'img-10', 'img-2']
    """
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', str(s))]


class ImageSequenceReader:
    """
    Reader for image sequences that mimics cv2.VideoCapture interface.

    Supports common image formats: .bmp, .png, .jpg, .jpeg, .tif, .tiff
    """

    SUPPORTED_EXTENSIONS = {'.bmp', '.png', '.jpg', '.jpeg', '.tif', '.tiff'}

    def __init__(self, folder_path: str):
        """
        Initialize the image sequence reader.

        Args:
            folder_path: Path to folder containing image sequence
        """
        self.folder_path = Path(folder_path)
        self.image_files = []
        self.current_index = 0
        self._is_opened = False

        if not self.folder_path.exists():
            print(f"Error: Folder does not exist: {folder_path}")
            return

        if not self.folder_path.is_dir():
            print(f"Error: Path is not a directory: {folder_path}")
            return

        # Find all image files
        for ext in self.SUPPORTED_EXTENSIONS:
            self.image_files.extend(self.folder_path.glob(f'*{ext}'))
            self.image_files.extend(self.folder_path.glob(f'*{ext.upper()}'))

        # Remove duplicates and sort naturally
        self.image_files = sorted(set(self.image_files), key=lambda p: natural_sort_key(p.name))

        if len(self.image_files) == 0:
            print(f"Error: No image files found in {folder_path}")
            return

        self._is_opened = True
        print(f"Found {len(self.image_files)} images in {folder_path}")

    def isOpened(self) -> bool:
        """Check if the reader is opened successfully."""
        return self._is_opened and len(self.image_files) > 0

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read the next frame.

        Returns:
            (success, frame) tuple, mimicking cv2.VideoCapture.read()
        """
        if not self._is_opened or self.current_index >= len(self.image_files):
            return False, None

        image_path = self.image_files[self.current_index]
        frame = cv2.imread(str(image_path))

        if frame is None:
            print(f"Warning: Could not read image: {image_path}")
            self.current_index += 1
            return False, None

        self.current_index += 1
        return True, frame

    def release(self):
        """Release resources (no-op for image sequence, but maintains interface)."""
        self._is_opened = False
        self.current_index = 0

    def get(self, prop_id: int) -> float:
        """
        Get property value (partial implementation for compatibility).

        Supported properties:
            cv2.CAP_PROP_FRAME_COUNT: Total number of frames
            cv2.CAP_PROP_POS_FRAMES: Current frame position
            cv2.CAP_PROP_FRAME_WIDTH: Frame width
            cv2.CAP_PROP_FRAME_HEIGHT: Frame height
        """
        if prop_id == cv2.CAP_PROP_FRAME_COUNT:
            return float(len(self.image_files))
        elif prop_id == cv2.CAP_PROP_POS_FRAMES:
            return float(self.current_index)
        elif prop_id in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT):
            if len(self.image_files) > 0:
                sample = cv2.imread(str(self.image_files[0]))
                if sample is not None:
                    if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
                        return float(sample.shape[1])
                    else:
                        return float(sample.shape[0])
        return 0.0

    def set(self, prop_id: int, value: float) -> bool:
        """
        Set property value (partial implementation for compatibility).

        Supported properties:
            cv2.CAP_PROP_POS_FRAMES: Set current frame position
        """
        if prop_id == cv2.CAP_PROP_POS_FRAMES:
            new_index = int(value)
            if 0 <= new_index < len(self.image_files):
                self.current_index = new_index
                return True
        return False

    def __len__(self) -> int:
        """Return total number of frames."""
        return len(self.image_files)

    def get_current_filename(self) -> str:
        """Get the filename of the current frame (useful for debugging)."""
        if 0 < self.current_index <= len(self.image_files):
            return self.image_files[self.current_index - 1].name
        return ""


def create_input_reader(input_path: str) -> Union[cv2.VideoCapture, ImageSequenceReader]:
    """
    Create appropriate reader based on input path.

    Args:
        input_path: Path to video file or image folder

    Returns:
        cv2.VideoCapture for video files, ImageSequenceReader for image folders
    """
    path = Path(input_path)

    if path.is_dir():
        # It's a directory - use image sequence reader
        return ImageSequenceReader(input_path)
    elif path.is_file():
        # It's a file - use video capture
        return cv2.VideoCapture(input_path)
    else:
        # Try as video file anyway (might be a URL or camera index)
        return cv2.VideoCapture(input_path)


# ============================================
# Default Configuration
# ============================================

DEFAULT_CONFIG = {
    # Output directories
    'output_dir': 'data/multiscale_ema_frames',
    'pyramid_dir': 'data/multiscale_ema_pyramid_frames',
    'debug_dir': 'data/debug_ema_frames',

    # EMA temporal accumulation
    'ema_alpha': 0.4,

    # Source scaling (upscale before optical flow)
    'source_scale': 1.5,

    # Feature detection (Shi-Tomasi) - adjusted for 1.5x scale
    'max_corners': 600,
    'quality_level': 0.01,
    'min_distance': 22,
    'block_size': 11,

    # Lucas-Kanade optical flow - adjusted for 1.5x scale
    'win_size': (29, 29),
    'max_level': 3,

    # RANSAC parameters
    'ransac_reproj_threshold': 4.5,

    # Downscale pyramid parameters
    'num_downscale_levels': 2,
    'downscale_targets': [1.0, 0.5],

    # Base parameters (for 1.0x scale)
    'base_threshold_sigma': 3.0,
    'base_morph_kernel_size': 3,
    'base_min_area': 100,
    'base_max_area': 50000,

    # Scale adjustment factors
    'upscale_threshold_factor': 0.85,
    'upscale_kernel_size': 3,
    'downscale_threshold_factor': 1.3,
    'downscale_kernel_increment': 2,

    # Valid mask parameters
    'border_margin': 30,

    # Frame sampling
    'frame_sample_interval': 1,

    # Track Coasting parameters
    'enable_tracking': True,
    'track_min_hits': 3,
    'track_max_coast_age': 10,
    'track_max_tentative_age': 3,
    'track_max_distance': 100,
    'track_min_iou': 0.1,
    'track_distance_weight': 0.6,
    'track_iou_weight': 0.4,
    'kalman_process_noise': 1.0,
    'kalman_measurement_noise': 1.0,
    'track_history_length': 15,

    # Multi-hypothesis thresholding
    'enable_multi_hypothesis': True,
    'multi_hypothesis_sigmas': [2.0, 2.5, 3.0, 3.5, 4.0],
    'hypothesis_merge_iou': 0.5,
    'sigma_weights': {2.0: 0.5, 2.5: 0.7, 3.0: 1.0, 3.5: 1.2, 4.0: 1.5},
}


class Config:
    """Configuration container with attribute access."""

    def __init__(self, **kwargs):
        config = DEFAULT_CONFIG.copy()
        config.update(kwargs)
        for key, value in config.items():
            setattr(self, key, value)

        # Build OpenCV parameters
        self.feature_params = dict(
            maxCorners=self.max_corners,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
            blockSize=self.block_size
        )

        self.lk_params = dict(
            winSize=self.win_size,
            maxLevel=self.max_level,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )


# ============================================
# Base Functions
# ============================================

def upscale_frame(frame: np.ndarray, scale: float) -> np.ndarray:
    """Upscale frame by given factor using INTER_CUBIC interpolation."""
    if scale == 1.0:
        return frame
    new_size = (int(frame.shape[1] * scale), int(frame.shape[0] * scale))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_CUBIC)


def downscale_to_target(image: np.ndarray, current_scale: float, target_scale: float) -> np.ndarray:
    """Downscale image from current_scale to target_scale."""
    if current_scale == target_scale:
        return image
    scale_ratio = target_scale / current_scale
    new_size = (int(image.shape[1] * scale_ratio), int(image.shape[0] * scale_ratio))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


class ResidualAccumulatorEMA:
    """Exponential Moving Average accumulator for residuals."""

    def __init__(self, alpha: float = 0.4):
        self.alpha = max(0.0, min(1.0, alpha))
        self.ema_residual = None
        self.frame_count = 0

    def add(self, residual: np.ndarray, valid_mask: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """Add a new residual and return EMA accumulated result."""
        self.frame_count += 1
        residual_float = residual.astype(np.float32)

        if self.ema_residual is None:
            self.ema_residual = residual_float
        else:
            self.ema_residual = (self.alpha * residual_float +
                                 (1.0 - self.alpha) * self.ema_residual)

        accumulated = np.clip(self.ema_residual, 0, 255).astype(np.uint8)
        return accumulated, valid_mask

    def reset(self):
        """Reset the accumulator."""
        self.ema_residual = None
        self.frame_count = 0

    def get_effective_frames(self) -> int:
        """Estimate effective number of frames contributing to EMA."""
        if self.alpha > 0:
            return min(self.frame_count, int(2.0 / self.alpha - 1))
        return self.frame_count


def detect_features(gray: np.ndarray, config: Config) -> np.ndarray:
    """Detect Shi-Tomasi features for tracking."""
    return cv2.goodFeaturesToTrack(gray, **config.feature_params)


def track_features(prev_gray: np.ndarray, curr_gray: np.ndarray,
                   prev_pts: np.ndarray, config: Config) -> Tuple:
    """Track features using Lucas-Kanade optical flow."""
    if prev_pts is None or len(prev_pts) == 0:
        return None, None, None

    curr_pts, status, err = cv2.calcOpticalFlowPyrLK(
        prev_gray, curr_gray, prev_pts, None, **config.lk_params
    )

    if status is not None:
        good_old = prev_pts[status.ravel() == 1]
        good_new = curr_pts[status.ravel() == 1]
        return good_old, good_new, status

    return None, None, None


def estimate_ego_motion_ransac(old_pts: np.ndarray, new_pts: np.ndarray,
                                config: Config) -> Tuple:
    """Estimate camera ego-motion using RANSAC homography."""
    if old_pts is None or len(old_pts) < 4:
        return None, None

    H, inlier_mask = cv2.findHomography(
        old_pts, new_pts, cv2.RANSAC, config.ransac_reproj_threshold
    )
    return H, inlier_mask


def warp_frame(frame: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Warp frame using homography."""
    h, w = frame.shape[:2]
    return cv2.warpPerspective(frame, H, (w, h))


def create_valid_warp_mask(warped_frame: np.ndarray, border_margin: int,
                           source_scale: float) -> np.ndarray:
    """Create mask of valid pixels after warping."""
    if len(warped_frame.shape) == 3:
        valid_mask = cv2.cvtColor(warped_frame, cv2.COLOR_BGR2GRAY) > 0
    else:
        valid_mask = warped_frame > 0

    valid_mask = valid_mask.astype(np.uint8) * 255

    erode_size = int(15 * source_scale)
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (erode_size, erode_size))
    valid_mask = cv2.erode(valid_mask, erode_kernel)

    if border_margin > 0:
        valid_mask[:border_margin, :] = 0
        valid_mask[-border_margin:, :] = 0
        valid_mask[:, :border_margin] = 0
        valid_mask[:, -border_margin:] = 0

    return valid_mask


def compute_residual(curr_gray: np.ndarray, warped_prev_gray: np.ndarray,
                     valid_mask: np.ndarray = None) -> np.ndarray:
    """Compute residual after ego-motion compensation."""
    residual = cv2.absdiff(curr_gray, warped_prev_gray)

    if valid_mask is not None:
        residual = cv2.bitwise_and(residual, residual, mask=valid_mask)

    return residual


# ============================================
# Track Coasting System
# ============================================

class KalmanTracker:
    """Constant-velocity Kalman filter for 2D object tracking."""

    def __init__(self, initial_centroid: Tuple[int, int],
                 process_noise: float = 1.0,
                 measurement_noise: float = 1.0):
        self.state = np.array([
            initial_centroid[0], initial_centroid[1], 0.0, 0.0
        ], dtype=np.float32)

        self.F = np.array([
            [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]
        ], dtype=np.float32)

        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        self.P = np.diag([10, 10, 100, 100]).astype(np.float32)
        self.Q = np.diag([1, 1, process_noise, process_noise]).astype(np.float32)
        self.R = np.diag([measurement_noise, measurement_noise]).astype(np.float32)

    def predict(self) -> Tuple[int, int]:
        """Predict next state."""
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.get_position()

    def update(self, measurement: Tuple[int, int]) -> Tuple[int, int]:
        """Update state with measurement."""
        z = np.array(measurement, dtype=np.float32)
        y = z - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y
        I = np.eye(4, dtype=np.float32)
        self.P = (I - K @ self.H) @ self.P
        return self.get_position()

    def get_position(self) -> Tuple[int, int]:
        """Return current estimated centroid position."""
        return (int(round(self.state[0])), int(round(self.state[1])))

    def get_velocity(self) -> Tuple[float, float]:
        """Return current estimated velocity."""
        return (float(self.state[2]), float(self.state[3]))

    def get_predicted_bbox(self, last_bbox: Tuple) -> Tuple:
        """Predict bounding box by translating last bbox by velocity."""
        x, y, w, h = last_bbox
        vx, vy = self.get_velocity()
        return (int(round(x + vx)), int(round(y + vy)), w, h)

    def get_speed(self) -> float:
        """Return current speed."""
        vx, vy = self.get_velocity()
        return np.sqrt(vx**2 + vy**2)


@dataclass
class Track:
    """Represents a tracked object with state history and lifecycle management."""

    track_id: int
    position_history: deque = field(default_factory=lambda: deque(maxlen=15))
    bbox_history: deque = field(default_factory=lambda: deque(maxlen=15))
    kalman: Optional[KalmanTracker] = None
    hits: int = 0
    misses: int = 0
    age: int = 0
    status: str = 'tentative'
    last_detection: Optional[Dict] = None
    last_detection_frame: int = 0
    confidence: float = 0.3
    preferred_scale: float = 1.0
    scale_hits: Dict[float, int] = field(default_factory=dict)

    def get_current_centroid(self) -> Tuple[int, int]:
        if self.position_history:
            return self.position_history[-1]
        return (0, 0)

    def get_current_bbox(self) -> Tuple[int, int, int, int]:
        if self.bbox_history:
            return self.bbox_history[-1]
        return (0, 0, 0, 0)


class TrackManager:
    """Manages the lifecycle of multiple object tracks."""

    def __init__(self, config: Config):
        self.tracks: List[Track] = []
        self.next_id: int = 1
        self.frame_count: int = 0

        self.min_hits = config.track_min_hits
        self.max_coast_age = config.track_max_coast_age
        self.max_tentative_age = config.track_max_tentative_age
        self.max_distance = config.track_max_distance
        self.min_iou = config.track_min_iou
        self.distance_weight = config.track_distance_weight
        self.iou_weight = config.track_iou_weight
        self.process_noise = config.kalman_process_noise
        self.measurement_noise = config.kalman_measurement_noise

    def step(self, detections: List[Dict]) -> List[Track]:
        """Process one frame of detections."""
        self.frame_count += 1

        for track in self.tracks:
            if track.kalman:
                track.kalman.predict()

        active_tracks = [t for t in self.tracks if t.status != 'deleted']
        matches, unmatched_tracks, unmatched_dets = self._associate(active_tracks, detections)

        for track_idx, det_idx in matches:
            self._update_track(active_tracks[track_idx], detections[det_idx])

        for track_idx in unmatched_tracks:
            self._coast_track(active_tracks[track_idx])

        for det_idx in unmatched_dets:
            self._create_track(detections[det_idx])

        self.tracks = [t for t in self.tracks if not self._should_delete(t)]

        return [t for t in self.tracks if t.status in ('confirmed', 'coasting')]

    def _compute_iou(self, bbox1: Tuple, bbox2: Tuple) -> float:
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)

        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        union_area = w1 * h1 + w2 * h2 - inter_area

        if union_area <= 0:
            return 0.0
        return inter_area / union_area

    def _compute_distance(self, c1: Tuple, c2: Tuple) -> float:
        return np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)

    def _associate(self, tracks: List[Track], detections: List[Dict]) -> Tuple:
        if len(tracks) == 0:
            return [], [], list(range(len(detections)))
        if len(detections) == 0:
            return [], list(range(len(tracks))), []

        num_tracks = len(tracks)
        num_dets = len(detections)

        cost_matrix = np.full((num_tracks, num_dets), 1e6, dtype=np.float32)
        valid_mask = np.zeros((num_tracks, num_dets), dtype=bool)

        for t, track in enumerate(tracks):
            if track.kalman:
                pred_centroid = track.kalman.get_position()
                if track.last_detection:
                    pred_bbox = track.kalman.get_predicted_bbox(track.last_detection['bbox'])
                else:
                    pred_bbox = track.get_current_bbox()
            else:
                pred_centroid = track.get_current_centroid()
                pred_bbox = track.get_current_bbox()

            for d, det in enumerate(detections):
                distance = self._compute_distance(pred_centroid, det['centroid'])
                iou = self._compute_iou(pred_bbox, det['bbox'])

                if distance <= self.max_distance and iou >= self.min_iou:
                    norm_distance = distance / self.max_distance
                    norm_iou_cost = 1.0 - iou
                    cost = self.distance_weight * norm_distance + self.iou_weight * norm_iou_cost
                    cost_matrix[t, d] = cost
                    valid_mask[t, d] = True

        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        matches = []
        unmatched_tracks = set(range(num_tracks))
        unmatched_dets = set(range(num_dets))

        for row, col in zip(row_indices, col_indices):
            if valid_mask[row, col]:
                matches.append((row, col))
                unmatched_tracks.discard(row)
                unmatched_dets.discard(col)

        return matches, list(unmatched_tracks), list(unmatched_dets)

    def _update_track(self, track: Track, detection: Dict):
        if track.kalman:
            track.kalman.update(detection['centroid'])

        track.position_history.append(detection['centroid'])
        track.bbox_history.append(detection['bbox'])
        track.hits += 1
        track.misses = 0
        track.age += 1
        track.last_detection = detection.copy()
        track.last_detection_frame = self.frame_count

        scale = detection.get('scale', 1.0)
        track.scale_hits[scale] = track.scale_hits.get(scale, 0) + 1
        if track.scale_hits:
            track.preferred_scale = max(track.scale_hits, key=track.scale_hits.get)

        if track.status == 'tentative' and track.hits >= self.min_hits:
            track.status = 'confirmed'
            track.confidence = 0.7
        elif track.status == 'coasting':
            track.status = 'confirmed'
            track.confidence = min(0.9, track.confidence + 0.2)
        else:
            track.confidence = min(0.95, track.confidence + 0.05)

    def _coast_track(self, track: Track):
        if track.kalman and track.last_detection:
            predicted_pos = track.kalman.get_position()
            predicted_bbox = track.kalman.get_predicted_bbox(track.last_detection['bbox'])
        else:
            predicted_pos = track.get_current_centroid()
            predicted_bbox = track.get_current_bbox()

        track.position_history.append(predicted_pos)
        track.bbox_history.append(predicted_bbox)
        track.misses += 1
        track.age += 1

        if track.status == 'confirmed' and track.misses == 1:
            track.status = 'coasting'
            track.confidence = max(0.3, track.confidence - 0.1)
        elif track.status == 'coasting':
            track.confidence = max(0.1, track.confidence - 0.1)

    def _create_track(self, detection: Dict):
        track = Track(
            track_id=self.next_id,
            hits=1, misses=0, age=1, status='tentative',
            last_detection=detection.copy(),
            last_detection_frame=self.frame_count,
            confidence=0.3,
            preferred_scale=detection.get('scale', 1.0)
        )

        track.kalman = KalmanTracker(
            detection['centroid'],
            self.process_noise,
            self.measurement_noise
        )

        track.position_history.append(detection['centroid'])
        track.bbox_history.append(detection['bbox'])
        track.scale_hits[detection.get('scale', 1.0)] = 1

        self.tracks.append(track)
        self.next_id += 1

    def _should_delete(self, track: Track) -> bool:
        if track.status == 'tentative' and track.age > self.max_tentative_age:
            return True
        if track.misses > self.max_coast_age:
            return True
        if track.confidence < 0.05:
            return True
        return False

    def get_statistics(self) -> Dict:
        confirmed = sum(1 for t in self.tracks if t.status == 'confirmed')
        coasting = sum(1 for t in self.tracks if t.status == 'coasting')
        tentative = sum(1 for t in self.tracks if t.status == 'tentative')

        return {
            'total_tracks': len(self.tracks),
            'confirmed': confirmed,
            'coasting': coasting,
            'tentative': tentative,
            'next_id': self.next_id,
            'frame_count': self.frame_count
        }


class MultiHypothesisDetector:
    """Run detection at multiple sigma thresholds and combine results."""

    def __init__(self, config: Config):
        self.sigmas = config.multi_hypothesis_sigmas
        self.sigma_weights = config.sigma_weights
        self.merge_iou = config.hypothesis_merge_iou

    def detect_multi_hypothesis(self, residual: np.ndarray, valid_mask: np.ndarray,
                                 kernel_size: int, min_area: int,
                                 max_area: int) -> Tuple[List[Dict], List[Dict]]:
        all_detections = []
        hypothesis_info = []

        for sigma in self.sigmas:
            motion_mask, thresh_val = adaptive_threshold_mad(residual, valid_mask, sigma)
            motion_mask = morphological_cleanup(motion_mask, kernel_size)
            detections = extract_bounding_boxes(motion_mask, min_area, max_area)

            weight = self.sigma_weights.get(sigma, 1.0)
            for det in detections:
                det['sigma'] = sigma
                det['threshold'] = thresh_val
                det['hypothesis_weight'] = weight

            all_detections.extend(detections)

            hypothesis_info.append({
                'sigma': sigma, 'threshold': thresh_val,
                'num_detections': len(detections), 'mask': motion_mask
            })

        merged_detections = self._merge_hypotheses(all_detections)
        return merged_detections, hypothesis_info

    def _compute_iou(self, bbox1: Tuple, bbox2: Tuple) -> float:
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)

        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        union_area = w1 * h1 + w2 * h2 - inter_area

        if union_area <= 0:
            return 0.0
        return inter_area / union_area

    def _merge_hypotheses(self, detections: List[Dict]) -> List[Dict]:
        if len(detections) == 0:
            return []

        clusters = []
        used = set()

        for i, det_i in enumerate(detections):
            if i in used:
                continue

            cluster = [det_i]
            used.add(i)

            for j, det_j in enumerate(detections):
                if j in used:
                    continue

                for det_c in cluster:
                    if self._compute_iou(det_c['bbox'], det_j['bbox']) > self.merge_iou:
                        cluster.append(det_j)
                        used.add(j)
                        break

            clusters.append(cluster)

        merged = []
        for cluster in clusters:
            sigmas_detected = sorted(set(d['sigma'] for d in cluster))
            num_hypotheses = len(sigmas_detected)
            total_weight = sum(d['hypothesis_weight'] for d in cluster)
            best_det = max(cluster, key=lambda d: d['hypothesis_weight'])

            max_possible_hypotheses = len(self.sigmas)
            max_possible_weight = sum(self.sigma_weights.get(s, 1.0) for s in self.sigmas)

            confidence = (
                0.4 * (num_hypotheses / max_possible_hypotheses) +
                0.4 * (total_weight / max_possible_weight) +
                0.2 * (best_det['sigma'] / max(self.sigmas))
            )
            confidence = min(1.0, confidence)

            merged.append({
                'bbox': best_det['bbox'],
                'centroid': best_det['centroid'],
                'area': best_det['area'],
                'scale': best_det.get('scale', 1.0),
                'confidence': confidence,
                'num_hypotheses': num_hypotheses,
                'sigmas_detected': sigmas_detected,
                'total_weight': total_weight
            })

        return merged


def tracks_to_detections(tracks: List[Track]) -> List[Dict]:
    """Convert track objects to detection dicts for output."""
    detections = []

    for track in tracks:
        det = {
            'bbox': track.get_current_bbox(),
            'centroid': track.get_current_centroid(),
            'area': track.last_detection['area'] if track.last_detection else 0,
            'scale': track.preferred_scale,
            'track_id': track.track_id,
            'track_status': track.status,
            'track_confidence': track.confidence,
            'track_hits': track.hits,
            'track_misses': track.misses,
            'track_age': track.age,
            'is_coasting': track.status == 'coasting'
        }

        if track.kalman:
            vx, vy = track.kalman.get_velocity()
            det['velocity'] = (vx, vy)
            det['speed'] = track.kalman.get_speed()

        detections.append(det)

    return detections


# ============================================
# Multi-Scale Pyramid Functions
# ============================================

def build_downscale_pyramid(image: np.ndarray, source_scale: float,
                            target_scales: List[float]) -> List[Tuple]:
    """Build downscale-only pyramid starting from source_scale."""
    pyramid = [(source_scale, image)]
    current_image = image
    current_scale = source_scale

    for target_scale in target_scales:
        if target_scale < current_scale:
            scaled_image = downscale_to_target(current_image, current_scale, target_scale)
            pyramid.append((target_scale, scaled_image))
            current_image = scaled_image
            current_scale = target_scale

    return pyramid


def get_scale_params(scale: float, config: Config) -> Dict:
    """Get scale-normalized parameters for any pyramid level."""
    if scale > 1.0:
        area_scale = scale ** 2
        kernel_size = config.upscale_kernel_size
        scale_steps = np.log2(scale)
        threshold_sigma = config.base_threshold_sigma * (config.upscale_threshold_factor ** scale_steps)
        min_area = int(config.base_min_area * area_scale)
        max_area = int(config.base_max_area * area_scale)
    elif scale < 1.0:
        level = int(np.log2(1.0 / scale))
        kernel_size = config.base_morph_kernel_size + (config.downscale_kernel_increment * level)
        threshold_sigma = config.base_threshold_sigma * (config.downscale_threshold_factor ** level)
        area_scale = scale ** 2
        min_area = max(10, int(config.base_min_area * area_scale))
        max_area = max(100, int(config.base_max_area * area_scale))
    else:
        kernel_size = config.base_morph_kernel_size
        threshold_sigma = config.base_threshold_sigma
        min_area = config.base_min_area
        max_area = config.base_max_area

    if kernel_size % 2 == 0:
        kernel_size += 1

    return {
        'scale': scale,
        'threshold_sigma': threshold_sigma,
        'kernel_size': kernel_size,
        'min_area': min_area,
        'max_area': max_area
    }


def adaptive_threshold_mad(residual: np.ndarray, valid_mask: np.ndarray = None,
                           sigma_multiplier: float = 3.0) -> Tuple[np.ndarray, float]:
    """Adaptive thresholding using Median Absolute Deviation (MAD)."""
    if valid_mask is not None:
        valid_pixels = residual[valid_mask > 0]
    else:
        valid_pixels = residual.ravel()

    if len(valid_pixels) == 0:
        return np.zeros_like(residual, dtype=np.uint8), 15.0

    median_val = np.median(valid_pixels)
    mad = np.median(np.abs(valid_pixels.astype(np.float32) - median_val))
    mad_std = mad * 1.4826

    threshold = median_val + sigma_multiplier * mad_std
    threshold = max(threshold, 15)

    _, motion_mask = cv2.threshold(residual, threshold, 255, cv2.THRESH_BINARY)

    if valid_mask is not None:
        motion_mask = cv2.bitwise_and(motion_mask, motion_mask, mask=valid_mask)

    return motion_mask.astype(np.uint8), threshold


def morphological_cleanup(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Apply morphological operations."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def extract_bounding_boxes(mask: np.ndarray, min_area: int = 100,
                           max_area: int = 50000) -> List[Dict]:
    """Extract bounding boxes using connected components."""
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

    detections = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]

        if min_area < area < max_area:
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            centroid = tuple(map(int, centroids[i]))

            detections.append({
                'bbox': (x, y, w, h),
                'centroid': centroid,
                'area': area
            })

    return detections


def process_pyramid_level(residual_scaled: np.ndarray, valid_mask: np.ndarray,
                          scale: float, config: Config,
                          multi_hypothesis: bool = False,
                          multi_hyp_detector: MultiHypothesisDetector = None) -> Tuple:
    """Process a single pyramid level with scale-appropriate parameters."""
    params = get_scale_params(scale, config)

    if valid_mask is not None:
        h, w = residual_scaled.shape[:2]
        valid_mask_scaled = cv2.resize(valid_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    else:
        valid_mask_scaled = None

    if multi_hypothesis and config.enable_multi_hypothesis and multi_hyp_detector:
        detections, hypothesis_info = multi_hyp_detector.detect_multi_hypothesis(
            residual_scaled, valid_mask_scaled,
            params['kernel_size'], params['min_area'], params['max_area']
        )

        base_info = next((h for h in hypothesis_info if h['sigma'] == 3.0),
                         hypothesis_info[0] if hypothesis_info else None)
        if base_info:
            motion_mask = base_info['mask']
            thresh_val = base_info['threshold']
        else:
            motion_mask = np.zeros_like(residual_scaled, dtype=np.uint8)
            thresh_val = 15.0
    else:
        motion_mask_raw, thresh_val = adaptive_threshold_mad(
            residual_scaled, valid_mask_scaled, params['threshold_sigma']
        )
        motion_mask = morphological_cleanup(motion_mask_raw, params['kernel_size'])
        detections = extract_bounding_boxes(motion_mask, params['min_area'], params['max_area'])

    return detections, motion_mask, thresh_val, params


def scale_detections_to_original(detections: List[Dict], scale: float) -> List[Dict]:
    """Scale detections from pyramid level coordinates to original image coordinates."""
    if scale == 1.0:
        for det in detections:
            det['scale'] = scale
        return detections

    inv_scale = 1.0 / scale
    scaled_detections = []

    for det in detections:
        x, y, w, h = det['bbox']
        cx, cy = det['centroid']

        scaled_det = {
            'bbox': (int(x * inv_scale), int(y * inv_scale),
                     int(w * inv_scale), int(h * inv_scale)),
            'centroid': (int(cx * inv_scale), int(cy * inv_scale)),
            'area': int(det['area'] * inv_scale * inv_scale),
            'scale': scale
        }

        for key in ['confidence', 'num_hypotheses', 'sigmas_detected']:
            if key in det:
                scaled_det[key] = det[key]

        scaled_detections.append(scaled_det)

    return scaled_detections


def merge_overlapping_detections(all_detections: List[Dict],
                                  iou_threshold: float = 0.3) -> List[Dict]:
    """Merge overlapping detections from different pyramid levels."""
    if len(all_detections) == 0:
        return []

    boxes = []
    for det in all_detections:
        x, y, w, h = det['bbox']
        boxes.append([x, y, x + w, y + h])

    boxes = np.array(boxes, dtype=np.float32)

    scores = []
    for det in all_detections:
        scale = det.get('scale', 1.0)
        scale_penalty = abs(np.log2(scale)) + 0.1
        base_score = det['area'] / scale_penalty

        if 'confidence' in det:
            base_score *= (1.0 + det['confidence'])

        scores.append(base_score)

    scores = np.array(scores, dtype=np.float32)

    indices = cv2.dnn.NMSBoxes(
        boxes.tolist(), scores.tolist(),
        score_threshold=0.0, nms_threshold=iou_threshold
    )

    if len(indices) > 0:
        if isinstance(indices[0], (list, np.ndarray)):
            indices = [i[0] for i in indices]
        merged = [all_detections[i] for i in indices]
    else:
        merged = []

    return merged


def process_multiscale_pyramid(residual: np.ndarray, valid_mask: np.ndarray,
                               config: Config,
                               multi_hypothesis: bool = False) -> Tuple[List[Dict], List[Dict]]:
    """Main multi-scale processing function."""
    pyramid = build_downscale_pyramid(residual, config.source_scale, config.downscale_targets)

    multi_hyp_detector = None
    if multi_hypothesis and config.enable_multi_hypothesis:
        multi_hyp_detector = MultiHypothesisDetector(config)

    all_detections = []
    level_info = []

    for scale, residual_scaled in pyramid:
        detections, mask, thresh, params = process_pyramid_level(
            residual_scaled, valid_mask, scale, config,
            multi_hypothesis=multi_hypothesis,
            multi_hyp_detector=multi_hyp_detector
        )

        scaled_detections = scale_detections_to_original(detections, scale)
        all_detections.extend(scaled_detections)

        level_info.append({
            'scale': scale,
            'shape': residual_scaled.shape,
            'params': params,
            'threshold': thresh,
            'detections': len(detections),
            'mask': mask
        })

    merged_detections = merge_overlapping_detections(all_detections)
    return merged_detections, level_info


# ============================================
# Visualization Functions
# ============================================

def get_track_status_color(status: str) -> Tuple[int, int, int]:
    """Get color based on track status."""
    colors = {
        'confirmed': (0, 255, 0),
        'coasting': (0, 255, 255),
        'tentative': (255, 0, 0),
    }
    return colors.get(status, (128, 128, 128))


def draw_dashed_line(img: np.ndarray, pt1: Tuple, pt2: Tuple,
                     color: Tuple, thickness: int = 1, dash_length: int = 10):
    """Draw a dashed line between two points."""
    dist = np.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)
    if dist < 1:
        return

    num_dashes = max(1, int(dist / dash_length))

    for i in range(0, num_dashes, 2):
        start_ratio = i / num_dashes
        end_ratio = min((i + 1) / num_dashes, 1.0)

        start = (int(pt1[0] + (pt2[0] - pt1[0]) * start_ratio),
                 int(pt1[1] + (pt2[1] - pt1[1]) * start_ratio))
        end = (int(pt1[0] + (pt2[0] - pt1[0]) * end_ratio),
               int(pt1[1] + (pt2[1] - pt1[1]) * end_ratio))

        cv2.line(img, start, end, color, thickness)


def draw_dashed_rectangle(img: np.ndarray, pt1: Tuple, pt2: Tuple,
                          color: Tuple, thickness: int = 1, dash_length: int = 10):
    """Draw a dashed rectangle."""
    x1, y1 = pt1
    x2, y2 = pt2

    draw_dashed_line(img, (x1, y1), (x2, y1), color, thickness, dash_length)
    draw_dashed_line(img, (x1, y2), (x2, y2), color, thickness, dash_length)
    draw_dashed_line(img, (x1, y1), (x1, y2), color, thickness, dash_length)
    draw_dashed_line(img, (x2, y1), (x2, y2), color, thickness, dash_length)


def draw_tracked_detections(frame: np.ndarray, tracked_detections: List[Dict],
                            show_velocity: bool = True) -> np.ndarray:
    """Draw tracked detections with status visualization."""
    output = frame.copy()

    for det in tracked_detections:
        x, y, w, h = det['bbox']
        cx, cy = det['centroid']

        track_id = det.get('track_id', -1)
        status = det.get('track_status', 'confirmed')
        confidence = det.get('track_confidence', 1.0)
        is_coasting = det.get('is_coasting', False)

        color = get_track_status_color(status)

        if is_coasting:
            draw_dashed_rectangle(output, (x, y), (x + w, y + h), color, 2, 8)
        else:
            cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)

        cv2.circle(output, (cx, cy), 5, (0, 0, 255), -1)

        if show_velocity and 'velocity' in det:
            vx, vy = det['velocity']
            scale_factor = 3.0
            end_x = int(cx + vx * scale_factor)
            end_y = int(cy + vy * scale_factor)
            cv2.arrowedLine(output, (cx, cy), (end_x, end_y), (255, 0, 255), 2, tipLength=0.3)

        label_parts = [f"ID:{track_id}", f"{confidence:.2f}"]
        if is_coasting:
            label_parts.append("[COAST]")
        label = " ".join(label_parts)

        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(output, (x, y - label_h - 8), (x + label_w + 4, y - 2), (0, 0, 0), -1)
        cv2.putText(output, label, (x + 2, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return output


def draw_track_statistics(frame: np.ndarray, track_manager: TrackManager,
                          position: str = 'top-right') -> np.ndarray:
    """Draw tracking statistics overlay on frame."""
    output = frame.copy()
    stats = track_manager.get_statistics()

    lines = [
        f"Tracks: {stats['total_tracks']}",
        f"  Confirmed: {stats['confirmed']}",
        f"  Coasting: {stats['coasting']}",
        f"  Tentative: {stats['tentative']}",
        f"Frame: {stats['frame_count']}"
    ]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    line_height = 20
    padding = 10

    max_width = max(cv2.getTextSize(line, font, font_scale, thickness)[0][0] for line in lines)
    total_height = len(lines) * line_height + padding * 2

    h, w = frame.shape[:2]

    if position == 'top-left':
        x, y = padding, padding
    elif position == 'top-right':
        x, y = w - max_width - padding * 2, padding
    elif position == 'bottom-left':
        x, y = padding, h - total_height - padding
    else:
        x, y = w - max_width - padding * 2, h - total_height - padding

    cv2.rectangle(output, (x, y), (x + max_width + padding * 2, y + total_height), (0, 0, 0), -1)
    cv2.rectangle(output, (x, y), (x + max_width + padding * 2, y + total_height), (255, 255, 255), 1)

    for i, line in enumerate(lines):
        text_y = y + padding + (i + 1) * line_height - 5
        cv2.putText(output, line, (x + padding, text_y), font, font_scale, (255, 255, 255), thickness)

    return output


# ============================================
# Main Processing Pipeline
# ============================================

def process_video(input_path: str, config: Config, debug: bool = False,
                   csv_output: str = None, save_frames: bool = False):
    """
    Process video or image sequence with multi-scale motion detection and track coasting.

    Args:
        input_path: Path to video file or folder containing image sequence
        config: Configuration object
        debug: If True, save intermediate processing steps (implies save_frames=True)
        csv_output: Path to output CSV file for detections (optional)
        save_frames: If True, save output frames with detection visualizations
    """

    # Debug implies save_frames
    if debug:
        save_frames = True

    # Create output directories only if saving frames
    if save_frames:
        for dir_path in [config.output_dir, config.pyramid_dir, config.debug_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

    # Create appropriate reader (video or image sequence)
    cap = create_input_reader(input_path)
    is_image_sequence = isinstance(cap, ImageSequenceReader)

    if not cap.isOpened():
        print(f"Error: Cannot open input {input_path}")
        return

    ret, prev_frame_orig = cap.read()
    if not ret:
        print("Error: Cannot read first frame")
        return

    orig_h, orig_w = prev_frame_orig.shape[:2]

    prev_frame_scaled = upscale_frame(prev_frame_orig, config.source_scale)
    prev_gray_scaled = cv2.cvtColor(prev_frame_scaled, cv2.COLOR_BGR2GRAY)
    prev_pts = detect_features(prev_gray_scaled, config)

    residual_accumulator = ResidualAccumulatorEMA(alpha=config.ema_alpha)

    track_manager = None
    if config.enable_tracking:
        track_manager = TrackManager(config)

    frame_count = 0
    saved_count = 0

    # Determine total frames for progress display
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_image_sequence else len(cap)

    # Print configuration
    print("=" * 60)
    print("MULTI-SCALE PYRAMID MOTION DETECTION")
    print("=" * 60)
    input_type = "Image sequence" if is_image_sequence else "Video"
    print(f"\n{input_type}: {input_path}")
    if is_image_sequence:
        print(f"Total frames: {total_frames}")
    print(f"Original frame size: {orig_w}x{orig_h}")
    print(f"Scaled frame size: {int(orig_w * config.source_scale)}x{int(orig_h * config.source_scale)}")
    print(f"Source scale: {config.source_scale}x")
    print(f"Downscale targets: {config.downscale_targets}")
    print(f"EMA alpha: {config.ema_alpha}")
    print(f"Tracking: {'ENABLED' if config.enable_tracking else 'DISABLED'}")
    print(f"Multi-hypothesis: {'ENABLED' if config.enable_multi_hypothesis else 'DISABLED'}")
    print(f"Save frames: {'ENABLED' if save_frames else 'DISABLED'}")
    if save_frames:
        print(f"  Output dir: {config.output_dir}")
    print(f"Debug mode: {debug}")
    if csv_output:
        print(f"CSV output: {csv_output}")
    print()

    # Initialize CSV output if requested
    csv_file = None
    csv_writer = None
    if csv_output:
        csv_path = Path(csv_output)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_file = open(csv_path, 'w', newline='', encoding='utf-8')
        csv_writer = csv.writer(csv_file)
        # Write header
        csv_writer.writerow([
            'frame_id', 'centroid_x', 'centroid_y',
            'bbox_x', 'bbox_y', 'bbox_width', 'bbox_height'
        ])

    scale_detection_counts = {}
    coast_events = 0
    total_tracked_frames = 0
    total_csv_rows = 0

    while True:
        ret, curr_frame_orig = cap.read()
        if not ret:
            break

        frame_count += 1

        curr_frame_scaled = upscale_frame(curr_frame_orig, config.source_scale)
        curr_gray_scaled = cv2.cvtColor(curr_frame_scaled, cv2.COLOR_BGR2GRAY)

        old_pts, new_pts, status = track_features(prev_gray_scaled, curr_gray_scaled, prev_pts, config)

        if old_pts is not None and len(old_pts) >= 4:
            H, inlier_mask = estimate_ego_motion_ransac(old_pts, new_pts, config)

            if H is not None:
                warped_prev_gray_scaled = warp_frame(prev_gray_scaled, H)
                valid_mask_scaled = create_valid_warp_mask(
                    warped_prev_gray_scaled, config.border_margin, config.source_scale
                )
                residual_scaled = compute_residual(
                    curr_gray_scaled, warped_prev_gray_scaled, valid_mask_scaled
                )

                accumulated_residual, accumulated_mask = residual_accumulator.add(
                    residual_scaled, valid_mask_scaled
                )

                detections, level_info = process_multiscale_pyramid(
                    accumulated_residual, accumulated_mask, config,
                    multi_hypothesis=config.enable_multi_hypothesis
                )

                if config.enable_tracking and track_manager is not None:
                    active_tracks = track_manager.step(detections)
                    output_detections = tracks_to_detections(active_tracks)

                    coasting_count = sum(1 for t in active_tracks if t.status == 'coasting')
                    if coasting_count > 0:
                        coast_events += coasting_count
                    total_tracked_frames += len(active_tracks)
                else:
                    output_detections = detections

                # Write detections to CSV if enabled
                if csv_writer is not None:
                    if len(output_detections) > 0:
                        for det in output_detections:
                            cx, cy = det['centroid']
                            bx, by, bw, bh = det['bbox']
                            csv_writer.writerow([
                                frame_count,  # frame_id (1-indexed)
                                cx,           # centroid_x
                                cy,           # centroid_y
                                bx,           # bbox_x
                                by,           # bbox_y
                                bw,           # bbox_width
                                bh            # bbox_height
                            ])
                            total_csv_rows += 1
                    else:
                        # No detections for this frame - write row with null values
                        csv_writer.writerow([
                            frame_count,  # frame_id (1-indexed)
                            '',           # centroid_x (null)
                            '',           # centroid_y (null)
                            '',           # bbox_x (null)
                            '',           # bbox_y (null)
                            '',           # bbox_width (null)
                            ''            # bbox_height (null)
                        ])
                        total_csv_rows += 1

                if frame_count % config.frame_sample_interval == 0:
                    # Save visualization frames if enabled
                    if save_frames:
                        if config.enable_tracking:
                            output_frame = draw_tracked_detections(curr_frame_orig, output_detections)
                            if track_manager is not None:
                                output_frame = draw_track_statistics(output_frame, track_manager, 'top-right')
                        else:
                            output_frame = curr_frame_orig.copy()
                            for det in output_detections:
                                x, y, w, h = det['bbox']
                                cv2.rectangle(output_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                                cv2.circle(output_frame, det['centroid'], 5, (0, 0, 255), -1)

                        cv2.imwrite(f"{config.output_dir}/frame_{saved_count:04d}.png", output_frame)
                        saved_count += 1

                    # Always collect statistics for logging
                    scale_counts = {}
                    for det in output_detections:
                        scale = det.get('scale', 1.0)
                        scale_key = f"{scale:.2f}x"
                        scale_counts[scale_key] = scale_counts.get(scale_key, 0) + 1
                        scale_detection_counts[scale_key] = scale_detection_counts.get(scale_key, 0) + 1

                    ema_info = f"[EMA eff~{residual_accumulator.get_effective_frames()}f]"

                    if config.enable_tracking and track_manager is not None:
                        stats = track_manager.get_statistics()
                        track_info = f"[Tracks: {stats['confirmed']}C/{stats['coasting']}P/{stats['tentative']}T]"
                    else:
                        track_info = ""

                    print(f"Frame {frame_count}: {len(output_detections)} outputs "
                          f"(by scale: {scale_counts}) {ema_info} {track_info}")

        if frame_count % 10 == 0 or (new_pts is not None and len(new_pts) < 50):
            prev_pts = detect_features(curr_gray_scaled, config)
        else:
            prev_pts = new_pts.reshape(-1, 1, 2) if new_pts is not None else detect_features(curr_gray_scaled, config)

        prev_frame_orig = curr_frame_orig.copy()
        prev_frame_scaled = curr_frame_scaled.copy()
        prev_gray_scaled = curr_gray_scaled.copy()

    cap.release()

    # Print final statistics
    print(f"\n{'=' * 60}")
    print("PROCESSING COMPLETE")
    print(f"{'=' * 60}")
    if save_frames:
        print(f"Saved {saved_count} frames to: {config.output_dir}")

    if config.enable_tracking and track_manager is not None:
        final_stats = track_manager.get_statistics()
        print(f"\nTracking Statistics:")
        print(f"  Total tracks created: {final_stats['next_id'] - 1}")
        print(f"  Final active tracks: {final_stats['total_tracks']}")
        print(f"  Coasting events: {coast_events}")
        if total_tracked_frames > 0:
            print(f"  Coast ratio: {coast_events / total_tracked_frames * 100:.1f}%")

    print(f"\nTotal detections by scale:")
    for scale, count in sorted(scale_detection_counts.items(),
                               key=lambda x: float(x[0].replace('x', '')), reverse=True):
        print(f"  {scale}: {count}")

    # Close CSV file and print stats
    if csv_file is not None:
        csv_file.close()
        print(f"\nCSV output saved to: {csv_output}")
        print(f"  Total rows written: {total_csv_rows}")


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description='Multi-Scale Pyramid Motion Detection with Track Coasting',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process and save only CSV output (no frame images)
    python motion_detection.py --input data/videos/I_BS_01/I_BS_01 --csv detections.csv

    # Save detection visualization frames
    python motion_detection.py --input data/videos/cars_1.mp4 --motion-frames

    # Save both CSV and visualization frames
    python motion_detection.py --input data/videos/I_BS_01/I_BS_01 --csv out.csv --motion-frames

    # Enable debug mode (saves intermediate processing steps)
    python motion_detection.py --input data/videos/cars_1.mp4 --debug

    # Disable tracking
    python motion_detection.py --input data/videos/I_BS_01/I_BS_01 --csv out.csv --no-tracking

Supported image formats for folders: .bmp, .png, .jpg, .jpeg, .tif, .tiff

CSV output format:
    frame_id,centroid_x,centroid_y,bbox_x,bbox_y,bbox_width,bbox_height
        """
    )

    # Required arguments
    parser.add_argument('--input', '-i', required=True,
                        help='Path to input video file or folder containing image sequence')

    # Output options
    parser.add_argument('--output-dir', '-o', default='data/multiscale_ema_frames',
                        help='Output directory for processed frames (default: data/multiscale_ema_frames)')
    parser.add_argument('--csv', '-c', default=None,
                        help='Output CSV file path for detections (columns: frame_id,centroid_x,centroid_y,bbox_x,bbox_y,bbox_width,bbox_height)')
    parser.add_argument('--motion-frames', '-m', action='store_true',
                        help='Save output frames with detection visualizations to output directory')
    parser.add_argument('--debug', '-d', action='store_true',
                        help='Enable debug mode to save intermediate processing steps (implies --motion-frames)')

    # Scale options
    parser.add_argument('--source-scale', type=float, default=1.5,
                        help='Source scale for optical flow (default: 1.5)')
    parser.add_argument('--downscale-targets', nargs='+', type=float, default=[1.0, 0.5],
                        help='Downscale targets for pyramid (default: 1.0 0.5)')

    # EMA options
    parser.add_argument('--ema-alpha', type=float, default=0.4,
                        help='EMA alpha weight for temporal smoothing (default: 0.4)')

    # Tracking options
    parser.add_argument('--no-tracking', action='store_true',
                        help='Disable track coasting')
    parser.add_argument('--no-multi-hypothesis', action='store_true',
                        help='Disable multi-hypothesis detection')
    parser.add_argument('--track-min-hits', type=int, default=3,
                        help='Minimum hits to confirm track (default: 3)')
    parser.add_argument('--track-max-coast', type=int, default=10,
                        help='Maximum frames to coast without detection (default: 10)')

    # Detection options
    parser.add_argument('--base-threshold-sigma', type=float, default=3.0,
                        help='Base threshold sigma for detection (default: 3.0)')
    parser.add_argument('--min-area', type=int, default=100,
                        help='Minimum detection area (default: 100)')
    parser.add_argument('--max-area', type=int, default=50000,
                        help='Maximum detection area (default: 50000)')

    args = parser.parse_args()

    # Build config from arguments
    config = Config(
        output_dir=args.output_dir,
        source_scale=args.source_scale,
        downscale_targets=args.downscale_targets,
        ema_alpha=args.ema_alpha,
        enable_tracking=not args.no_tracking,
        enable_multi_hypothesis=not args.no_multi_hypothesis,
        track_min_hits=args.track_min_hits,
        track_max_coast_age=args.track_max_coast,
        base_threshold_sigma=args.base_threshold_sigma,
        base_min_area=args.min_area,
        base_max_area=args.max_area,
    )

    # Process input (video or image sequence)
    process_video(args.input, config, debug=args.debug, csv_output=args.csv,
                  save_frames=args.motion_frames)


if __name__ == '__main__':
    main()