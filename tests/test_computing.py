"""Unit tests for dyslexia-app/eye_tracking/computing.py.

The module lives outside the installed package, so we add its parent directory
to sys.path.  pyautogui (a GUI library) is mocked before any import so that
tests can run in headless CI environments.
"""
import sys
import os
from unittest.mock import MagicMock

# Add dyslexia-app to sys.path so that `from utils import ...` resolves.
_APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dyslexia-app"))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

# Mock pyautogui before utils (and transitively computing) is imported.
_pyautogui_mock = MagicMock()
_pyautogui_mock.size.return_value = (1920, 1080)
sys.modules.setdefault("pyautogui", _pyautogui_mock)

import math
import numpy as np
import pytest

from eye_tracking.computing import compute_scale, gaze_to_angles, fit_map, apply_map


class TestComputeScale:
    def test_two_known_points(self):
        pts = np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]])
        assert abs(compute_scale(pts) - 5.0) < 1e-9

    def test_single_point_returns_one(self):
        pts = np.array([[1.0, 2.0, 3.0]])
        assert compute_scale(pts) == 1.0

    def test_identical_points_returns_zero(self):
        pts = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
        assert compute_scale(pts) == 0.0

    def test_symmetric_result(self):
        pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        # 3 pairs: (0,1)=1, (0,2)=1, (1,2)=sqrt(2) → mean ≈ 1.138
        expected = (1 + 1 + math.sqrt(2)) / 3
        assert abs(compute_scale(pts) - expected) < 1e-9

    def test_returns_float(self):
        pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        assert isinstance(compute_scale(pts), float)


class TestGazeToAngles:
    def test_straight_ahead_zero_angles(self):
        d = np.array([0.0, 0.0, -1.0])
        yaw, pitch = gaze_to_angles(d)
        assert abs(yaw) < 1e-5
        assert abs(pitch) < 1e-5

    def test_looking_right_negative_yaw(self):
        # Camera convention: d[0]>0 is rightward; the final sign flip returns -yaw.
        d = np.array([1.0, 0.0, -1.0])
        yaw, _ = gaze_to_angles(d)
        assert yaw < 0

    def test_looking_left_positive_yaw(self):
        d = np.array([-1.0, 0.0, -1.0])
        yaw, _ = gaze_to_angles(d)
        assert yaw > 0

    def test_looking_down_negative_pitch(self):
        # Camera convention: d[1]>0 is downward; pitch is negated when d[1]>0.
        d = np.array([0.0, 1.0, -1.0])
        _, pitch = gaze_to_angles(d)
        assert pitch < 0

    def test_looking_up_positive_pitch(self):
        d = np.array([0.0, -1.0, -1.0])
        _, pitch = gaze_to_angles(d)
        assert pitch > 0

    def test_unnormalised_input_handled(self):
        d_unit = np.array([0.0, 0.0, -1.0])
        d_scaled = d_unit * 5.0
        yaw1, pitch1 = gaze_to_angles(d_unit)
        yaw2, pitch2 = gaze_to_angles(d_scaled)
        assert abs(yaw1 - yaw2) < 1e-5
        assert abs(pitch1 - pitch2) < 1e-5

    def test_returns_degrees_not_radians(self):
        d = np.array([1.0, 0.0, -1.0])
        yaw, _ = gaze_to_angles(d)
        assert abs(yaw) > 1.0  # would be ~0.785 if in radians, ~45 in degrees


class TestFitMap:
    """Tests for the least-squares calibration fit."""

    def _grid_samples(self, scale=100, offset_x=960, offset_y=540):
        """9-point grid where screen_x = yaw*scale + offset_x, etc."""
        angles = [(-9, -5), (0, -5), (9, -5),
                  (-9, 0),  (0, 0),  (9, 0),
                  (-9, 5),  (0, 5),  (9, 5)]
        return [(y, p, int(y * scale + offset_x), int(p * scale + offset_y))
                for y, p in angles]

    def test_returns_two_coefficient_vectors(self):
        cx, cy = fit_map(self._grid_samples())
        assert len(cx) == 3 and len(cy) == 3

    def test_perfect_linear_data_recovers_coefficients(self):
        samples = self._grid_samples(scale=100, offset_x=960, offset_y=540)
        cx, cy = fit_map(samples)
        # cx[0] ≈ 100, cx[1] ≈ 0, cx[2] ≈ 960
        assert abs(cx[0] - 100) < 1.0
        assert abs(cx[2] - 960) < 2.0
        assert abs(cy[1] - 100) < 1.0
        assert abs(cy[2] - 540) < 2.0

    def test_minimum_nine_samples_required(self):
        samples = self._grid_samples()
        assert len(samples) == 9
        cx, cy = fit_map(samples)
        assert cx is not None and cy is not None


class TestApplyMap:
    """Tests for the calibrated gaze-to-screen mapping."""

    MONITOR_W = 1920
    MONITOR_H = 1080

    def _identity_coeffs(self):
        cx = np.array([100.0, 0.0, 960.0])
        cy = np.array([0.0, 100.0, 540.0])
        return cx, cy

    def test_applies_linear_map(self):
        cx, cy = self._identity_coeffs()
        x, y = apply_map(1.0, 1.0, cx, cy)
        assert x == 1060
        assert y == 640

    def test_clips_large_values_to_upper_bound(self):
        cx, cy = self._identity_coeffs()
        x, y = apply_map(10000.0, 10000.0, cx, cy)
        assert x == self.MONITOR_W - 10
        assert y == self.MONITOR_H - 10

    def test_clips_small_values_to_lower_bound(self):
        cx, cy = self._identity_coeffs()
        x, y = apply_map(-10000.0, -10000.0, cx, cy)
        assert x == 10
        assert y == 10

    def test_returns_integers(self):
        cx, cy = self._identity_coeffs()
        x, y = apply_map(0.0, 0.0, cx, cy)
        assert isinstance(x, int) and isinstance(y, int)

    def test_zero_angles_map_to_screen_center(self):
        cx = np.array([100.0, 0.0, 960.0])
        cy = np.array([0.0, 100.0, 540.0])
        x, y = apply_map(0.0, 0.0, cx, cy)
        assert x == 960
        assert y == 540
