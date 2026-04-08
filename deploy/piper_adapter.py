#!/usr/bin/env python3
"""
AgileX PiPER adapter for Xiaomi-Robotics-0 action chunks.

This module bridges:
  Xiaomi model output: [dx, dy, dz, droll, dpitch, dyaw, gripper]
to
  PiPER backend end-pose + gripper commands.

Note: For LIBERO-style policies, the rotation delta is commonly represented as an axis-angle
vector (so(3) rotation vector). This adapter supports both Euler-delta and axis-angle-delta
rotation updates (see `PiperSDKAdapter.rotation_delta_mode`).

It uses the CALVIN-style task id by default:
  task_id = "calvin_abcd_orig"

References:
  - https://github.com/agilexrobotics/piper_sdk
  - https://github.com/agilexrobotics/piper_ros
  - https://github.com/Reimagine-Robotics/piper_control
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import threading
import time
from dataclasses import dataclass
from typing import Any, Literal, Optional

import numpy as np
from PIL import Image

from deploy.client import Client as XiaomiModelClient
from deploy.piper_backend_support import (
    PIPER_BACKEND_CHOICES,
    PiperBackendName,
    build_piper_control_interface,
    build_piper_sdk_interface,
    has_sdk_only_options,
    resolve_piper_backend,
)
from deploy.piper_support import (
    PiperDiagnosticReport,
    collect_piper_diagnostic_report,
    format_piper_diagnostic_report,
    summarize_motion_blockers,
)


LOGGER = logging.getLogger(__name__)

INSTALLATION_POS_CODES: dict[str, int] = {
    "upright": 0x01,
    "left": 0x02,
    "right": 0x03,
}


def _try_set_sdk_installation_pos(arm: Any, installation_pos: str) -> bool:
    code = INSTALLATION_POS_CODES[installation_pos]
    motion_ctrl = getattr(arm, "MotionCtrl_2", None)
    if motion_ctrl is None:
        return False

    try:
        motion_ctrl(0x01, 0x01, 0, 0, 0, code)
    except TypeError:
        return False
    time.sleep(0.1)
    return True


def _arm_enable_feedback_ok(arm: Any) -> Optional[bool]:
    if hasattr(arm, "GetArmEnableStatus"):
        try:
            enabled = arm.GetArmEnableStatus()
        except Exception:
            enabled = None
        if enabled is not None and len(enabled) >= 6:
            return all(bool(x) for x in enabled[:6])

    report = collect_piper_diagnostic_report(arm)
    if report.enable_status:
        return all(bool(x) for x in report.enable_status[:6])
    if report.driver_disabled:
        return False
    return None


def _force_enable_sdk_arm(arm: Any, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if hasattr(arm, "EnableArm"):
            enabled_any_joint = False
            for joint_index in range(1, 7):
                try:
                    arm.EnableArm(joint_index)
                    enabled_any_joint = True
                except TypeError:
                    try:
                        arm.EnableArm(joint_index, 0x02)
                        enabled_any_joint = True
                    except TypeError:
                        continue

            if not enabled_any_joint:
                try:
                    arm.EnableArm(7, 0x02)
                except TypeError:
                    try:
                        arm.EnableArm(7)
                    except TypeError:
                        for joint_index in range(1, 7):
                            arm.EnableArm(joint_index)
        elif hasattr(arm, "EnablePiper"):
            arm.EnablePiper()

        enabled = _arm_enable_feedback_ok(arm)
        if enabled is None:
            time.sleep(0.25)
            return
        if enabled:
            return
        time.sleep(0.05)

    raise RuntimeError(
        "PiPER enable timeout. Check CAN wiring/power and try again.\n"
        f"{format_piper_diagnostic_report(collect_piper_diagnostic_report(arm))}"
    )


def _ensure_arm_enabled_or_raise(arm: Any, backend_name: str) -> None:
    enabled = _arm_enable_feedback_ok(arm)
    if enabled is False:
        raise RuntimeError(
            f"PiPER {backend_name} backend finished setup with arm drivers still disabled.\n"
            f"{format_piper_diagnostic_report(collect_piper_diagnostic_report(arm))}"
        )


def _wait_for_piper_control_ready(
    robot: Any, piper_interface: Any, timeout_s: float
) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            arm_enabled = bool(robot.is_arm_enabled())
            control_mode = robot.control_mode
            arm_status = robot.arm_status
            teach_status = robot.teach_status
        except Exception:
            time.sleep(0.1)
            continue

        if (
            arm_enabled
            and control_mode == piper_interface.ControlMode.CAN_COMMAND
            and arm_status == piper_interface.ArmStatus.NORMAL
            and teach_status == piper_interface.TeachStatus.OFF
        ):
            return
        time.sleep(0.1)

    raise RuntimeError(
        "PiPER piper_control backend did not become ready after reset/mode switch.\n"
        f"{format_piper_diagnostic_report(collect_piper_diagnostic_report(robot.piper))}"
    )


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _wrap_pi(rad: float) -> float:
    return (rad + math.pi) % (2.0 * math.pi) - math.pi


def _normalize_instruction(text: str) -> str:
    text = text.strip()
    if not text:
        return "Move to target."
    text = text.rstrip(".")
    return text[0].upper() + text[1:] + "."


def _center_crop_keep_size(image: Image.Image, crop_ratio: float = 0.95) -> Image.Image:
    w, h = image.size
    cw = int(w * crop_ratio)
    ch = int(h * crop_ratio)
    left = (w - cw) // 2
    top = (h - ch) // 2
    out = image.crop((left, top, left + cw, top + ch))
    return out.resize((w, h), Image.Resampling.BILINEAR)


def _as_numpy(x) -> np.ndarray:
    if hasattr(x, "cpu"):
        return x.cpu().numpy()
    return np.asarray(x)


def hash_data_to_seed(data: dict[str, Any], max_bytes: int = 4) -> int:
    """Compute a stable seed for model inputs.

    Mirrors the hashing approach used by the benchmark eval clients in this repo.
    """

    def custom_encoder(obj: Any):
        if isinstance(obj, np.ndarray):
            return {
                "__type__": "numpy",
                "dtype": str(obj.dtype),
                "shape": obj.shape,
                "data": obj.tobytes().hex(),
            }

        if isinstance(obj, Image.Image):
            img_hash = hashlib.md5(obj.tobytes()).hexdigest()
            return {
                "__type__": "PIL.Image",
                "mode": obj.mode,
                "size": obj.size,
                "content_hash": img_hash,
            }

        if isinstance(obj, set):
            return sorted(list(obj))

        raise TypeError(f"Type {type(obj)} is not JSON serializable")

    json_str = json.dumps(
        data,
        default=custom_encoder,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    hex_hash = hashlib.sha256(json_str.encode("utf-8")).hexdigest()

    seed_int = int(hex_hash, 16)
    if max_bytes > 0:
        seed_int = seed_int % (2 ** (8 * max_bytes))
    return seed_int


PROFILE_DEFAULT_TASK_ID: dict[str, str] = {
    "calvin": "calvin_abcd_orig",
    "libero": "libero_all",
}


def _normalize_profile(profile: str) -> Literal["calvin", "libero"]:
    value = profile.strip().lower()
    if value not in PROFILE_DEFAULT_TASK_ID:
        expected = ", ".join(sorted(PROFILE_DEFAULT_TASK_ID))
        raise ValueError(
            f"Unsupported profile '{profile}'. Expected one of: {expected}."
        )
    return value  # type: ignore[return-value]


def _euler_xyz_to_quat_xyzw(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert XYZ Euler angles (rad) to quaternion (x, y, z, w)."""
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)

    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    qw = cr * cp * cy + sr * sp * sy
    return np.array([qx, qy, qz, qw], dtype=np.float64)


def _axis_angle_to_quat_xyzw(axis_angle: np.ndarray) -> np.ndarray:
    """Convert axis-angle vector to quaternion (x, y, z, w)."""
    vec = np.asarray(axis_angle, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(vec))
    if theta <= 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    axis = vec / theta
    half = 0.5 * theta
    s = math.sin(half)
    return np.array(
        [axis[0] * s, axis[1] * s, axis[2] * s, math.cos(half)], dtype=np.float64
    )


def _quat_mul_xyzw(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product for quaternions (x, y, z, w)."""
    a = np.asarray(q1, dtype=np.float64).reshape(4)
    b = np.asarray(q2, dtype=np.float64).reshape(4)
    x1, y1, z1, w1 = (float(a[0]), float(a[1]), float(a[2]), float(a[3]))
    x2, y2, z2, w2 = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))

    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    return np.array([x, y, z, w], dtype=np.float64)


def _quat_xyzw_to_euler_xyz(quat_xyzw: np.ndarray) -> np.ndarray:
    """Convert quaternion (x, y, z, w) to XYZ Euler angles (rad)."""
    quat = np.asarray(quat_xyzw, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(quat))
    if norm <= 1e-12:
        return np.zeros(3, dtype=np.float64)
    quat = quat / norm
    x, y, z, w = (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    sinp = _clip(sinp, -1.0, 1.0)
    pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw], dtype=np.float64)


def _quat_xyzw_to_axis_angle(quat_xyzw: np.ndarray) -> np.ndarray:
    """Convert quaternion (x, y, z, w) to axis-angle vector."""
    quat = np.asarray(quat_xyzw, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(quat))
    if norm <= 1e-12:
        return np.zeros(3, dtype=np.float64)

    quat = quat / norm
    w = _clip(float(quat[3]), -1.0, 1.0)
    den = math.sqrt(max(0.0, 1.0 - w * w))
    if math.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float64)
    return quat[:3] * (2.0 * math.acos(w) / den)


def _open_camera(index: int, width: int, height: int, fps: int):
    import cv2

    backend = getattr(cv2, "CAP_V4L2", 0)
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera index {index}.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def _bgr_frame_to_pil(frame_bgr: np.ndarray) -> Image.Image:
    import cv2

    return Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))


class _OpenCvImagePairReader:
    def __init__(
        self,
        base_index: int,
        wrist_index: int,
        width: int,
        height: int,
        fps: int,
    ) -> None:
        self._cap_base = _open_camera(base_index, width, height, fps)
        self._shared_capture = int(base_index) == int(wrist_index)
        if self._shared_capture:
            self._cap_wrist = self._cap_base
        else:
            try:
                self._cap_wrist = _open_camera(wrist_index, width, height, fps)
            except Exception:
                self._cap_base.release()
                raise

    @staticmethod
    def _read_frame(cap, timeout_s: float, label: str) -> np.ndarray:
        deadline = time.time() + max(0.01, timeout_s)
        while time.time() < deadline:
            ok, frame = cap.read()
            if ok and frame is not None:
                return frame
            time.sleep(0.01)

        raise TimeoutError(f"Timed out waiting for OpenCV {label} frame.")

    def get_latest_pil(self, timeout_s: float) -> tuple[Image.Image, Image.Image]:
        base_bgr = self._read_frame(self._cap_base, timeout_s=timeout_s, label="base")
        wrist_bgr = (
            base_bgr.copy()
            if self._shared_capture
            else self._read_frame(self._cap_wrist, timeout_s=timeout_s, label="wrist")
        )
        return _bgr_frame_to_pil(base_bgr), _bgr_frame_to_pil(wrist_bgr)

    def close(self) -> None:
        self._cap_base.release()
        if not self._shared_capture:
            self._cap_wrist.release()


class _RosImagePairSubscriber:
    def __init__(
        self,
        base_topic: str,
        wrist_topic: str,
        sync_queue: int,
        sync_slop: float,
        node_name: str,
    ) -> None:
        import message_filters
        import rospy
        from cv_bridge import CvBridge
        from sensor_msgs.msg import Image as RosImage

        if not rospy.core.is_initialized():
            rospy.init_node(node_name, anonymous=True, disable_signals=True)

        self._rospy = rospy
        self._bridge = CvBridge()
        self._lock = threading.Lock()
        self._latest = None

        self._base_sub = message_filters.Subscriber(base_topic, RosImage)
        self._wrist_sub = message_filters.Subscriber(wrist_topic, RosImage)
        self._sync = message_filters.ApproximateTimeSynchronizer(
            [self._base_sub, self._wrist_sub],
            queue_size=max(1, sync_queue),
            slop=max(0.0, sync_slop),
        )
        self._sync.registerCallback(self._callback)

    def _callback(self, base_msg, wrist_msg) -> None:
        try:
            base_bgr = self._bridge.imgmsg_to_cv2(base_msg, desired_encoding="bgr8")
            wrist_bgr = self._bridge.imgmsg_to_cv2(wrist_msg, desired_encoding="bgr8")
        except Exception:
            return

        with self._lock:
            self._latest = (base_bgr, wrist_bgr)

    def get_latest_pil(self, timeout_s: float) -> tuple[Image.Image, Image.Image]:
        deadline = time.time() + max(0.01, timeout_s)
        while time.time() < deadline:
            with self._lock:
                latest = self._latest

            if latest is not None:
                base_bgr, wrist_bgr = latest
                return _bgr_frame_to_pil(base_bgr), _bgr_frame_to_pil(wrist_bgr)

            if self._rospy.is_shutdown():
                raise RuntimeError("ROS node is shut down.")

            time.sleep(0.01)

        raise TimeoutError("Timed out waiting for synced ROS camera frames.")

    def close(self) -> None:
        return


@dataclass
class PiperSafety:
    x_min_m: float = -0.40
    x_max_m: float = 0.40
    y_min_m: float = -0.35
    y_max_m: float = 0.35
    z_min_m: float = 0.05
    z_max_m: float = 0.45
    resync_position_threshold_m: float = 0.08
    resync_rotation_threshold_rad: float = 0.75
    resync_gripper_threshold_m: float = 0.03
    gripper_command_epsilon_m: float = 0.002
    gripper_retry_interval_s: float = 1.0


class PiperAdapterBase:
    """Translate model delta actions into PiPER motion commands."""

    backend_name: PiperBackendName = "piper_sdk"

    def __init__(
        self,
        move_speed_percent: int = 30,
        linear_step_m: float = 0.01,
        angular_step_rad: float = 0.12,
        rotation_delta_mode: Literal["euler_xyz", "axis_angle"] = "euler_xyz",
        gripper_open_m: float = 0.08,
        gripper_close_m: float = 0.0,
        gripper_effort: int = 1000,
        gripper_threshold: float = 0.0,
        gripper_mode: Literal["binary", "continuous"] = "binary",
        safety: Optional[PiperSafety] = None,
    ) -> None:
        self.move_speed_percent = int(_clip(move_speed_percent, 1, 100))
        self.linear_step_m = float(linear_step_m)
        self.angular_step_rad = float(angular_step_rad)
        if rotation_delta_mode not in {"euler_xyz", "axis_angle"}:
            raise ValueError(
                f"Unsupported rotation_delta_mode '{rotation_delta_mode}'. Expected 'euler_xyz' or 'axis_angle'."
            )
        self.rotation_delta_mode = rotation_delta_mode
        self.gripper_open_m = float(gripper_open_m)
        self.gripper_close_m = float(gripper_close_m)
        self.gripper_effort = int(_clip(gripper_effort, 0, 5000))
        self.gripper_threshold = float(gripper_threshold)
        if gripper_mode not in {"binary", "continuous"}:
            raise ValueError(
                f"Unsupported gripper_mode '{gripper_mode}'. Expected 'binary' or 'continuous'."
            )
        self.gripper_mode = gripper_mode
        self.safety = safety if safety is not None else PiperSafety()

        self._target_pose = np.zeros(6, dtype=np.float64)
        self._target_gripper_m = self.gripper_close_m
        self._last_motion_diag_at = 0.0
        self._last_motion_diag_key: tuple[str, ...] = ()
        self._last_gripper_command_m: Optional[float] = None
        self._last_gripper_command_at = 0.0

    def connect(self, enable_timeout_s: float = 5.0) -> None:
        raise NotImplementedError

    def get_diagnostic_report(self) -> PiperDiagnosticReport:
        raise NotImplementedError

    def read_end_pose_m_rad(self) -> np.ndarray:
        raise NotImplementedError

    def read_gripper_m(self) -> float:
        raise NotImplementedError

    def _send_pose_command(self, pose6_m_rad: np.ndarray) -> None:
        raise NotImplementedError

    def _send_gripper_command(self, gripper_m: float) -> None:
        raise NotImplementedError

    def _prepare_motion_command(self, motion_requested: bool) -> None:
        self._maybe_warn_motion_blocked(motion_requested)

    def _sync_targets_from_feedback(self) -> None:
        self._target_pose = self.read_end_pose_m_rad()
        self._target_gripper_m = self.read_gripper_m()
        self._last_gripper_command_m = self._target_gripper_m
        self._last_gripper_command_at = time.time()

    def _maybe_resync_targets_from_feedback(self) -> None:
        try:
            actual_pose = self.read_end_pose_m_rad()
            actual_gripper = self.read_gripper_m()
        except Exception as exc:
            LOGGER.warning(
                "Failed to read PiPER feedback before applying action; keeping previous target: %s",
                exc,
            )
            return

        pos_error = float(np.max(np.abs(actual_pose[:3] - self._target_pose[:3])))
        rot_error = float(
            np.max(
                np.abs(
                    [
                        _wrap_pi(float(actual_pose[idx] - self._target_pose[idx]))
                        for idx in range(3, 6)
                    ]
                )
            )
        )
        gripper_error = abs(float(actual_gripper) - float(self._target_gripper_m))

        if (
            pos_error <= self.safety.resync_position_threshold_m
            and rot_error <= self.safety.resync_rotation_threshold_rad
        ):
            return

        LOGGER.warning(
            "Resyncing PiPER target to live feedback for safety (%s): position_error=%.3fm, rotation_error=%.3frad, gripper_error=%.3fm",
            self.backend_name,
            pos_error,
            rot_error,
            gripper_error,
        )
        self._target_pose = actual_pose

    def _should_send_gripper_command(self, gripper_m: float) -> bool:
        now = time.time()
        if self._last_gripper_command_m is None:
            return True
        if (
            abs(float(gripper_m) - float(self._last_gripper_command_m))
            > self.safety.gripper_command_epsilon_m
        ):
            return True
        if now - self._last_gripper_command_at < self.safety.gripper_retry_interval_s:
            return False
        try:
            actual_gripper = self.read_gripper_m()
        except Exception:
            return True
        return (
            abs(float(actual_gripper) - float(gripper_m))
            > self.safety.resync_gripper_threshold_m
        )

    def _maybe_warn_motion_blocked(self, motion_requested: bool) -> None:
        if not motion_requested:
            return

        now = time.time()
        if now - self._last_motion_diag_at < 1.0:
            return

        report = self.get_diagnostic_report()
        blockers = summarize_motion_blockers(report)
        if not blockers:
            self._last_motion_diag_at = now
            self._last_motion_diag_key = ()
            return

        if (
            blockers == self._last_motion_diag_key
            and now - self._last_motion_diag_at < 3.0
        ):
            return

        self._last_motion_diag_at = now
        self._last_motion_diag_key = blockers
        LOGGER.warning(
            "PiPER arm motion may be blocked (%s): %s\n%s",
            self.backend_name,
            "; ".join(blockers),
            format_piper_diagnostic_report(report),
        )

    def build_calvin_state32(self) -> np.ndarray:
        pose = self.read_end_pose_m_rad()
        gripper = self.read_gripper_m()
        state7 = np.array(
            [pose[0], pose[1], pose[2], pose[3], pose[4], pose[5], gripper],
            dtype=np.float32,
        )
        return np.concatenate([state7, np.zeros(25, dtype=np.float32)], axis=0)

    def build_libero_state32(self) -> np.ndarray:
        pose = self.read_end_pose_m_rad()
        gripper_width = self.read_gripper_m()

        quat_xyzw = _euler_xyz_to_quat_xyzw(pose[3], pose[4], pose[5])
        axis_angle = _quat_xyzw_to_axis_angle(quat_xyzw)

        finger_qpos = 0.5 * gripper_width
        state8 = np.array(
            [
                pose[0],
                pose[1],
                pose[2],
                axis_angle[0],
                axis_angle[1],
                axis_angle[2],
                finger_qpos,
                finger_qpos,
            ],
            dtype=np.float32,
        )
        return np.concatenate([state8, np.zeros(24, dtype=np.float32)], axis=0)

    def apply_delta_action(self, action7: np.ndarray) -> None:
        a = np.asarray(action7, dtype=np.float64).reshape(-1)
        if a.size < 7:
            raise ValueError(f"Expected at least 7 values, got {a.size}")

        self._maybe_resync_targets_from_feedback()
        prev_target_pose = self._target_pose.copy()
        delta_xyz = np.clip(a[:3], -1.0, 1.0) * self.linear_step_m
        delta_rot = np.clip(a[3:6], -1.0, 1.0) * self.angular_step_rad

        self._target_pose[:3] = self._target_pose[:3] + delta_xyz

        if self.rotation_delta_mode == "axis_angle":
            current_quat = _euler_xyz_to_quat_xyzw(
                float(self._target_pose[3]),
                float(self._target_pose[4]),
                float(self._target_pose[5]),
            )
            delta_quat = _axis_angle_to_quat_xyzw(delta_rot)
            new_quat = _quat_mul_xyzw(delta_quat, current_quat)
            self._target_pose[3:] = _quat_xyzw_to_euler_xyz(new_quat)
        else:
            self._target_pose[3:] = self._target_pose[3:] + delta_rot

        self._target_pose[0] = _clip(
            self._target_pose[0], self.safety.x_min_m, self.safety.x_max_m
        )
        self._target_pose[1] = _clip(
            self._target_pose[1], self.safety.y_min_m, self.safety.y_max_m
        )
        self._target_pose[2] = _clip(
            self._target_pose[2], self.safety.z_min_m, self.safety.z_max_m
        )
        self._target_pose[3] = _wrap_pi(self._target_pose[3])
        self._target_pose[4] = _wrap_pi(self._target_pose[4])
        self._target_pose[5] = _wrap_pi(self._target_pose[5])

        if self.gripper_mode == "continuous":
            # LIBERO convention: dim 6 is a continuous normalized gripper command
            # in [-1, 1].  Map linearly: -1 → gripper_close_m, +1 → gripper_open_m.
            t = (float(np.clip(a[6], -1.0, 1.0)) + 1.0) * 0.5
            self._target_gripper_m = self.gripper_close_m + t * (
                self.gripper_open_m - self.gripper_close_m
            )
        else:
            # CALVIN convention: binarize to fully open / fully closed.
            self._target_gripper_m = (
                self.gripper_open_m
                if a[6] > self.gripper_threshold
                else self.gripper_close_m
            )
        self._target_gripper_m = _clip(
            self._target_gripper_m,
            min(self.gripper_close_m, self.gripper_open_m),
            max(self.gripper_close_m, self.gripper_open_m),
        )

        motion_requested = bool(
            np.linalg.norm(self._target_pose - prev_target_pose) > 1e-9
        )
        self._send_pose_and_gripper(
            self._target_pose,
            self._target_gripper_m,
            motion_requested=motion_requested,
        )

    def _send_pose_and_gripper(
        self, pose6_m_rad: np.ndarray, gripper_m: float, motion_requested: bool = True
    ) -> None:
        self._prepare_motion_command(motion_requested)

        try:
            self._send_pose_command(pose6_m_rad)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to send PiPER {self.backend_name} pose command.\n"
                f"{format_piper_diagnostic_report(self.get_diagnostic_report())}"
            ) from exc

        if self._should_send_gripper_command(gripper_m):
            try:
                self._send_gripper_command(gripper_m)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to send PiPER {self.backend_name} gripper command.\n"
                    f"{format_piper_diagnostic_report(self.get_diagnostic_report())}"
                ) from exc
            self._last_gripper_command_m = float(gripper_m)
            self._last_gripper_command_at = time.time()

    def execute_chunk(
        self, action_chunk: np.ndarray, replan_steps: int = 4, dt_s: float = 0.08
    ) -> None:
        chunk = np.asarray(action_chunk)
        if chunk.ndim != 2 or chunk.shape[1] < 7:
            raise ValueError(f"Expected shape [T, >=7], got {chunk.shape}")

        steps = int(min(replan_steps, chunk.shape[0]))
        for i in range(steps):
            self.apply_delta_action(chunk[i, :7])
            time.sleep(dt_s)


class PiperSDKAdapter(PiperAdapterBase):
    """Translate model delta actions into PiPER SDK commands."""

    backend_name: PiperBackendName = "piper_sdk"

    def __init__(
        self,
        can_port: str = "can0",
        move_speed_percent: int = 30,
        linear_step_m: float = 0.01,
        angular_step_rad: float = 0.12,
        rotation_delta_mode: Literal["euler_xyz", "axis_angle"] = "euler_xyz",
        gripper_open_m: float = 0.08,
        gripper_close_m: float = 0.0,
        gripper_effort: int = 1000,
        gripper_threshold: float = 0.0,
        gripper_mode: Literal["binary", "continuous"] = "binary",
        safety: Optional[PiperSafety] = None,
        judge_flag: bool = True,
        dh_is_offset: Optional[int] = None,
        sdk_joint_limit: Optional[bool] = None,
        sdk_gripper_limit: Optional[bool] = None,
        force_slave_mode: bool = False,
        installation_pos: Literal["upright", "left", "right"] = "upright",
    ) -> None:
        self.arm = build_piper_sdk_interface(
            can_port=can_port,
            judge_flag=judge_flag,
            dh_is_offset=dh_is_offset,
            sdk_joint_limit=sdk_joint_limit,
            sdk_gripper_limit=sdk_gripper_limit,
        )
        self.force_slave_mode = bool(force_slave_mode)
        self.installation_pos = installation_pos
        super().__init__(
            move_speed_percent=move_speed_percent,
            linear_step_m=linear_step_m,
            angular_step_rad=angular_step_rad,
            rotation_delta_mode=rotation_delta_mode,
            gripper_open_m=gripper_open_m,
            gripper_close_m=gripper_close_m,
            gripper_effort=gripper_effort,
            gripper_threshold=gripper_threshold,
            gripper_mode=gripper_mode,
            safety=safety,
        )

    def connect(self, enable_timeout_s: float = 5.0) -> None:
        self.arm.ConnectPort()
        time.sleep(0.2)
        if not _try_set_sdk_installation_pos(self.arm, self.installation_pos):
            LOGGER.warning(
                "PiPER SDK backend could not confirm installation position '%s'; continuing without explicit install-pose config.",
                self.installation_pos,
            )
        if self.force_slave_mode and hasattr(self.arm, "MasterSlaveConfig"):
            self.arm.MasterSlaveConfig(0xFC, 0, 0, 0)
            time.sleep(0.1)
        self._enable_arm(enable_timeout_s)
        self._set_pose_mode()
        self.arm.GripperCtrl(0, self.gripper_effort, 0x01, 0)
        self._sync_targets_from_feedback()
        _ensure_arm_enabled_or_raise(self.arm, self.backend_name)

    def _enable_arm(self, timeout_s: float) -> None:
        _force_enable_sdk_arm(self.arm, timeout_s)

    def _set_pose_mode(self) -> None:
        if hasattr(self.arm, "ModeCtrl"):
            self.arm.ModeCtrl(0x01, 0x00, self.move_speed_percent, 0x00)
            return
        if hasattr(self.arm, "MotionCtrl_2"):
            self.arm.MotionCtrl_2(0x01, 0x00, self.move_speed_percent, 0x00)
            return
        raise RuntimeError("No supported mode control API found in piper_sdk.")

    def get_diagnostic_report(self) -> PiperDiagnosticReport:
        return collect_piper_diagnostic_report(self.arm)

    def read_end_pose_m_rad(self) -> np.ndarray:
        msg = self.arm.GetArmEndPoseMsgs()
        ep = msg.end_pose
        x = float(ep.X_axis) / 1_000_000.0
        y = float(ep.Y_axis) / 1_000_000.0
        z = float(ep.Z_axis) / 1_000_000.0
        rx = math.radians(float(ep.RX_axis) / 1000.0)
        ry = math.radians(float(ep.RY_axis) / 1000.0)
        rz = math.radians(float(ep.RZ_axis) / 1000.0)
        return np.array([x, y, z, rx, ry, rz], dtype=np.float64)

    def read_gripper_m(self) -> float:
        msg = self.arm.GetArmGripperMsgs()
        return float(msg.gripper_state.grippers_angle) / 1_000_000.0

    def _prepare_motion_command(self, motion_requested: bool) -> None:
        super()._prepare_motion_command(motion_requested)
        self._set_pose_mode()

    def _send_pose_command(self, pose6_m_rad: np.ndarray) -> None:
        x = int(round(float(pose6_m_rad[0]) * 1_000_000.0))
        y = int(round(float(pose6_m_rad[1]) * 1_000_000.0))
        z = int(round(float(pose6_m_rad[2]) * 1_000_000.0))
        rx = int(round(math.degrees(float(pose6_m_rad[3])) * 1000.0))
        ry = int(round(math.degrees(float(pose6_m_rad[4])) * 1000.0))
        rz = int(round(math.degrees(float(pose6_m_rad[5])) * 1000.0))
        self.arm.EndPoseCtrl(x, y, z, rx, ry, rz)

    def _send_gripper_command(self, gripper_m: float) -> None:
        gripper_val = int(round(abs(float(gripper_m)) * 1_000_000.0))
        self.arm.GripperCtrl(gripper_val, self.gripper_effort, 0x01, 0)


class PiperControlAdapter(PiperAdapterBase):
    """Translate model delta actions into piper_control commands."""

    backend_name: PiperBackendName = "piper_control"

    def __init__(
        self,
        can_port: str = "can0",
        move_speed_percent: int = 30,
        linear_step_m: float = 0.01,
        angular_step_rad: float = 0.12,
        rotation_delta_mode: Literal["euler_xyz", "axis_angle"] = "euler_xyz",
        gripper_open_m: float = 0.08,
        gripper_close_m: float = 0.0,
        gripper_effort: int = 1000,
        gripper_threshold: float = 0.0,
        gripper_mode: Literal["binary", "continuous"] = "binary",
        safety: Optional[PiperSafety] = None,
        judge_flag: bool = True,
        dh_is_offset: Optional[int] = None,
        sdk_joint_limit: Optional[bool] = None,
        sdk_gripper_limit: Optional[bool] = None,
        force_slave_mode: bool = False,
        installation_pos: Literal["upright", "left", "right"] = "upright",
        piper_control_src: Optional[str] = None,
    ) -> None:
        if has_sdk_only_options(
            judge_flag=judge_flag,
            dh_is_offset=dh_is_offset,
            sdk_joint_limit=sdk_joint_limit,
            sdk_gripper_limit=sdk_gripper_limit,
            force_slave_mode=force_slave_mode,
        ):
            unsupported = []
            if not judge_flag:
                unsupported.append("judge_flag=False")
            if dh_is_offset is not None:
                unsupported.append(f"dh_is_offset={dh_is_offset}")
            if sdk_joint_limit is not None:
                unsupported.append(f"sdk_joint_limit={sdk_joint_limit}")
            if sdk_gripper_limit is not None:
                unsupported.append(f"sdk_gripper_limit={sdk_gripper_limit}")
            if force_slave_mode:
                unsupported.append("force_slave_mode=True")
            raise ValueError(
                f"piper_control backend does not support: {', '.join(unsupported)}. "
                "Use --piper-backend=piper_sdk or --piper-backend=auto to select a backend "
                "that honors these options."
            )
        self.robot, modules = build_piper_control_interface(
            can_port=can_port,
            piper_control_src=piper_control_src,
        )
        self._piper_init = modules.piper_init
        self._piper_interface = modules.piper_interface
        self.arm = self.robot.piper
        self.installation_pos = installation_pos
        gripper_open_m = min(float(gripper_open_m), float(self.robot.gripper_angle_max))
        gripper_effort = min(
            int(gripper_effort),
            int(round(float(self.robot.gripper_effort_max) * 1000.0)),
        )
        super().__init__(
            move_speed_percent=move_speed_percent,
            linear_step_m=linear_step_m,
            angular_step_rad=angular_step_rad,
            rotation_delta_mode=rotation_delta_mode,
            gripper_open_m=gripper_open_m,
            gripper_close_m=gripper_close_m,
            gripper_effort=gripper_effort,
            gripper_threshold=gripper_threshold,
            gripper_mode=gripper_mode,
            safety=safety,
        )

    def connect(self, enable_timeout_s: float = 5.0) -> None:
        try:
            self.robot.set_installation_pos(
                self._piper_interface.ArmInstallationPos.from_string(
                    self.installation_pos
                )
            )

            LOGGER.warning(
                "Resetting PiPER arm/gripper via piper_control reset flow; the arm may depower briefly during reset_arm()."
            )
            self._piper_init.reset_arm(
                self.robot,
                arm_controller=self._piper_interface.ArmController.POSITION_VELOCITY,
                move_mode=self._piper_interface.MoveMode.JOINT,
                timeout_seconds=enable_timeout_s,
            )
            self._piper_init.reset_gripper(
                self.robot,
                timeout_seconds=enable_timeout_s,
            )
            self._set_pose_mode()
            _wait_for_piper_control_ready(
                self.robot,
                self._piper_interface,
                timeout_s=max(enable_timeout_s, 2.0),
            )
            self._sync_targets_from_feedback()
            _ensure_arm_enabled_or_raise(self.arm, self.backend_name)
        except Exception as exc:
            raise RuntimeError(
                "Failed to initialize PiPER via piper_control.\n"
                f"{format_piper_diagnostic_report(self.get_diagnostic_report())}"
            ) from exc

    def _set_pose_mode(self) -> None:
        self.robot.set_arm_mode(
            speed=self.move_speed_percent,
            move_mode=self._piper_interface.MoveMode.POSITION,
            arm_controller=self._piper_interface.ArmController.POSITION_VELOCITY,
        )

    def get_diagnostic_report(self) -> PiperDiagnosticReport:
        return collect_piper_diagnostic_report(self.arm)

    def read_end_pose_m_rad(self) -> np.ndarray:
        return np.asarray(self.robot.get_end_effector_pose(), dtype=np.float64)

    def read_gripper_m(self) -> float:
        position_m, _ = self.robot.get_gripper_state()
        return float(position_m)

    def _prepare_motion_command(self, motion_requested: bool) -> None:
        super()._prepare_motion_command(motion_requested)
        self._set_pose_mode()

    def _send_pose_command(self, pose6_m_rad: np.ndarray) -> None:
        self.robot.command_cartesian_position(pose6_m_rad.tolist())

    def _send_gripper_command(self, gripper_m: float) -> None:
        self.robot.command_gripper(
            position=abs(float(gripper_m)),
            effort=self.gripper_effort / 1000.0,
        )


def _build_piper_adapter(
    backend: str,
    can_port: str,
    rotation_delta_mode: Literal["euler_xyz", "axis_angle"],
    gripper_mode: Literal["binary", "continuous"],
    judge_flag: bool,
    dh_is_offset: Optional[int],
    sdk_joint_limit: Optional[bool],
    sdk_gripper_limit: Optional[bool],
    force_slave_mode: bool,
    installation_pos: Literal["upright", "left", "right"],
    piper_control_src: Optional[str],
) -> tuple[PiperBackendName, PiperAdapterBase]:
    sdk_only = has_sdk_only_options(
        judge_flag=judge_flag,
        dh_is_offset=dh_is_offset,
        sdk_joint_limit=sdk_joint_limit,
        sdk_gripper_limit=sdk_gripper_limit,
        force_slave_mode=force_slave_mode,
    )
    selected_backend = resolve_piper_backend(
        backend,
        piper_control_src,
        sdk_only_options_requested=sdk_only,
    )
    if selected_backend == "piper_control":
        return selected_backend, PiperControlAdapter(
            can_port=can_port,
            rotation_delta_mode=rotation_delta_mode,
            gripper_mode=gripper_mode,
            judge_flag=judge_flag,
            dh_is_offset=dh_is_offset,
            sdk_joint_limit=sdk_joint_limit,
            sdk_gripper_limit=sdk_gripper_limit,
            force_slave_mode=force_slave_mode,
            installation_pos=installation_pos,
            piper_control_src=piper_control_src,
        )
    return selected_backend, PiperSDKAdapter(
        can_port=can_port,
        rotation_delta_mode=rotation_delta_mode,
        gripper_mode=gripper_mode,
        judge_flag=judge_flag,
        dh_is_offset=dh_is_offset,
        sdk_joint_limit=sdk_joint_limit,
        sdk_gripper_limit=sdk_gripper_limit,
        force_slave_mode=force_slave_mode,
        installation_pos=installation_pos,
    )


class XiaomiPiperController:
    """Model-server -> PiPER adapter bridge."""

    def __init__(
        self,
        model_host: str,
        model_port: int,
        piper_adapter: PiperAdapterBase,
        profile: Literal["calvin", "libero"] = "calvin",
        task_id: Optional[str] = None,
    ) -> None:
        self.model = XiaomiModelClient(host=model_host, port=model_port)
        self.piper = piper_adapter
        self.profile = _normalize_profile(profile)
        self.task_id = (
            task_id if task_id is not None else PROFILE_DEFAULT_TASK_ID[self.profile]
        )

    def _build_state32(self) -> np.ndarray:
        if self.profile == "libero":
            return self.piper.build_libero_state32()
        return self.piper.build_calvin_state32()

    def _postprocess_gripper(self, chunk: np.ndarray) -> np.ndarray:
        if chunk.ndim != 2 or chunk.shape[1] < 7:
            raise ValueError(f"Expected [T, >=7] action chunk, got {chunk.shape}")

        chunk = chunk.copy()
        if self.profile == "calvin":
            # Match eval_calvin behavior.
            chunk[:, 6] = np.where(chunk[:, 6] > 0.0, 1.0, -1.0)
        else:
            # Keep LIBERO gripper output continuous in [-1, 1].
            chunk[:, 6] = np.clip(chunk[:, 6], -1.0, 1.0)
        return chunk

    def infer_action_chunk(
        self, base_img: Image.Image, wrist_img: Image.Image, instruction: str
    ) -> np.ndarray:
        base_img = _center_crop_keep_size(base_img.convert("RGB"), crop_ratio=0.95)
        wrist_img = _center_crop_keep_size(wrist_img.convert("RGB"), crop_ratio=0.95)

        model_inputs: dict[str, Any] = {
            "task_id": self.task_id,
            "state": self._build_state32(),
            "base": base_img,
            "wrist_left": wrist_img,
            "language": _normalize_instruction(instruction),
        }
        model_inputs["seed"] = hash_data_to_seed(model_inputs)

        raw = self.model(**model_inputs)
        arr = _as_numpy(raw)

        if arr.ndim != 3:
            raise ValueError(f"Unexpected model output shape: {arr.shape}")
        return arr[0, :, :7].astype(np.float32)

    def run_once(
        self,
        base_img: Image.Image,
        wrist_img: Image.Image,
        instruction: str,
        replan_steps: int = 4,
        dt_s: float = 0.08,
    ) -> None:
        chunk = self._postprocess_gripper(
            self.infer_action_chunk(base_img, wrist_img, instruction)
        )
        self.piper.execute_chunk(chunk, replan_steps=replan_steps, dt_s=dt_s)

    def close(self) -> None:
        self.model.close()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Single-step Xiaomi-Robotics-0 -> PiPER adapter demo"
    )
    p.add_argument("--model-host", default="127.0.0.1")
    p.add_argument("--model-port", type=int, default=10086)
    p.add_argument("--can-port", default="can0")
    p.add_argument(
        "--piper-backend",
        choices=PIPER_BACKEND_CHOICES,
        default="auto",
        help="PiPER control backend. 'auto' prefers piper_sdk when SDK-only options are set, otherwise prefers piper_control when importable.",
    )
    p.add_argument(
        "--piper-control-src",
        default=None,
        help="Optional piper_control checkout root or src dir used when the package is not installed.",
    )
    p.add_argument(
        "--judge-flag",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="SDK backend only: pass judge_flag to piper_sdk when supported.",
    )
    p.add_argument(
        "--dh-is-offset",
        type=int,
        choices=(0, 1),
        default=None,
        help="SDK backend only: override piper_sdk dh_is_offset when supported. Firmware S-V1.6-3+ typically needs 1.",
    )
    p.add_argument(
        "--sdk-joint-limit",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="SDK backend only: override piper_sdk start_sdk_joint_limit when supported.",
    )
    p.add_argument(
        "--sdk-gripper-limit",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="SDK backend only: override piper_sdk start_sdk_gripper_limit when supported.",
    )
    p.add_argument(
        "--force-slave-mode",
        action="store_true",
        help="SDK backend only: force MasterSlaveConfig(0xFC,0,0,0) during connect when supported.",
    )
    p.add_argument(
        "--installation-pos",
        choices=sorted(INSTALLATION_POS_CODES),
        default="upright",
        help="Arm installation pose for the mounted robot; change this for side-mounted arms.",
    )
    p.add_argument(
        "--profile",
        choices=sorted(PROFILE_DEFAULT_TASK_ID),
        default="calvin",
        help="Model profile for state packing and gripper postprocess.",
    )
    p.add_argument(
        "--task-id",
        default=None,
        help="Optional override. Defaults to profile task id: calvin_abcd_orig/libero_all.",
    )
    p.add_argument("--instruction", required=True)
    p.add_argument("--base-image", default=None, help="Path to RGB base camera image")
    p.add_argument("--wrist-image", default=None, help="Path to RGB wrist camera image")
    p.add_argument(
        "--base-cam-index",
        type=int,
        default=None,
        help="OpenCV camera index for base view",
    )
    p.add_argument(
        "--wrist-cam-index",
        type=int,
        default=None,
        help="OpenCV camera index for wrist view",
    )
    p.add_argument("--camera-width", type=int, default=640)
    p.add_argument("--camera-height", type=int, default=480)
    p.add_argument("--camera-fps", type=int, default=30)
    p.add_argument(
        "--camera-frame-timeout",
        type=float,
        default=0.5,
        help="Timeout in seconds when reading OpenCV camera frames",
    )
    p.add_argument(
        "--ros-camera-mode",
        action="store_true",
        help="Use ROS image topics instead of OpenCV device indices",
    )
    p.add_argument("--ros-base-topic", default="/camera/base/image_raw")
    p.add_argument("--ros-wrist-topic", default="/camera/wrist/image_raw")
    p.add_argument("--ros-sync-queue", type=int, default=10)
    p.add_argument("--ros-sync-slop", type=float, default=0.05)
    p.add_argument("--ros-frame-timeout", type=float, default=2.0)
    p.add_argument("--ros-node-name", default="xiaomi_piper_adapter")
    p.add_argument(
        "--max-loops", type=int, default=0, help="Live mode loops; 0 means infinite"
    )
    p.add_argument(
        "--loop-sleep",
        type=float,
        default=0.0,
        help="Optional sleep between replans in live mode",
    )
    p.add_argument("--replan-steps", type=int, default=4)
    p.add_argument("--dt", type=float, default=0.08)
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    image_mode = args.base_image is not None or args.wrist_image is not None
    live_mode = args.base_cam_index is not None
    ros_mode = bool(args.ros_camera_mode)

    mode_count = int(image_mode) + int(live_mode) + int(ros_mode)
    if mode_count != 1:
        raise ValueError(
            "Choose exactly one input mode: image files, OpenCV cameras, or --ros-camera-mode."
        )
    if image_mode and (args.base_image is None or args.wrist_image is None):
        raise ValueError(
            "Image-path mode requires both --base-image and --wrist-image."
        )

    rotation_delta_mode: Literal["euler_xyz", "axis_angle"] = (
        "axis_angle" if args.profile == "libero" else "euler_xyz"
    )
    gripper_mode: Literal["binary", "continuous"] = (
        "continuous" if args.profile == "libero" else "binary"
    )
    selected_backend, adapter = _build_piper_adapter(
        backend=args.piper_backend,
        can_port=args.can_port,
        rotation_delta_mode=rotation_delta_mode,
        gripper_mode=gripper_mode,
        judge_flag=args.judge_flag,
        dh_is_offset=args.dh_is_offset,
        sdk_joint_limit=args.sdk_joint_limit,
        sdk_gripper_limit=args.sdk_gripper_limit,
        force_slave_mode=args.force_slave_mode,
        installation_pos=args.installation_pos,
        piper_control_src=args.piper_control_src,
    )
    print(f"[INFO] Using PiPER backend: {selected_backend}")
    try:
        adapter.connect()
    except Exception as exc:
        if args.piper_backend == "auto" and selected_backend == "piper_control":
            LOGGER.warning(
                "PiPER piper_control backend failed to initialize cleanly; falling back to piper_sdk: %s",
                exc,
            )
            selected_backend, adapter = _build_piper_adapter(
                backend="piper_sdk",
                can_port=args.can_port,
                rotation_delta_mode=rotation_delta_mode,
                gripper_mode=gripper_mode,
                judge_flag=args.judge_flag,
                dh_is_offset=args.dh_is_offset,
                sdk_joint_limit=args.sdk_joint_limit,
                sdk_gripper_limit=args.sdk_gripper_limit,
                force_slave_mode=args.force_slave_mode,
                installation_pos=args.installation_pos,
                piper_control_src=args.piper_control_src,
            )
            print(f"[INFO] Falling back to PiPER backend: {selected_backend}")
            adapter.connect()
        else:
            raise

    bridge = XiaomiPiperController(
        model_host=args.model_host,
        model_port=args.model_port,
        piper_adapter=adapter,
        profile=args.profile,
        task_id=args.task_id,
    )

    try:
        if image_mode:
            base_img = Image.open(args.base_image).convert("RGB")
            wrist_img = Image.open(args.wrist_image).convert("RGB")
            bridge.run_once(
                base_img=base_img,
                wrist_img=wrist_img,
                instruction=args.instruction,
                replan_steps=args.replan_steps,
                dt_s=args.dt,
            )
            return

        loops = 0

        if ros_mode:
            wrist_topic = args.ros_wrist_topic
            if not wrist_topic:
                wrist_topic = args.ros_base_topic
                print(
                    "[WARN] --ros-wrist-topic not provided. Reusing base topic as wrist view."
                )

            ros_images = _RosImagePairSubscriber(
                base_topic=args.ros_base_topic,
                wrist_topic=wrist_topic,
                sync_queue=args.ros_sync_queue,
                sync_slop=args.ros_sync_slop,
                node_name=args.ros_node_name,
            )
            try:
                while args.max_loops <= 0 or loops < args.max_loops:
                    base_img, wrist_img = ros_images.get_latest_pil(
                        timeout_s=args.ros_frame_timeout
                    )
                    bridge.run_once(
                        base_img=base_img,
                        wrist_img=wrist_img,
                        instruction=args.instruction,
                        replan_steps=args.replan_steps,
                        dt_s=args.dt,
                    )

                    loops += 1
                    if args.loop_sleep > 0:
                        time.sleep(args.loop_sleep)
            finally:
                ros_images.close()
            return

        wrist_index = (
            args.wrist_cam_index
            if args.wrist_cam_index is not None
            else args.base_cam_index
        )
        if args.wrist_cam_index is None:
            print(
                "[WARN] --wrist-cam-index not provided. Reusing base camera as wrist view."
            )

        opencv_images = _OpenCvImagePairReader(
            base_index=args.base_cam_index,
            wrist_index=wrist_index,
            width=args.camera_width,
            height=args.camera_height,
            fps=args.camera_fps,
        )
        try:
            while args.max_loops <= 0 or loops < args.max_loops:
                try:
                    base_img, wrist_img = opencv_images.get_latest_pil(
                        timeout_s=args.camera_frame_timeout
                    )
                except TimeoutError:
                    time.sleep(0.01)
                    continue

                bridge.run_once(
                    base_img=base_img,
                    wrist_img=wrist_img,
                    instruction=args.instruction,
                    replan_steps=args.replan_steps,
                    dt_s=args.dt,
                )

                loops += 1
                if args.loop_sleep > 0:
                    time.sleep(args.loop_sleep)
        finally:
            opencv_images.close()

    finally:
        bridge.close()


if __name__ == "__main__":
    main()
