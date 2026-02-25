#!/usr/bin/env python3
"""
AgileX PiPER adapter for Xiaomi-Robotics-0 action chunks.

This module bridges:
  Xiaomi model output: [dx, dy, dz, droll, dpitch, dyaw, gripper]
to
  PiPER SDK end-pose + gripper commands.

It uses the CALVIN-style task id by default:
  task_id = "calvin_abcd_orig"

References:
  - https://github.com/agilexrobotics/piper_sdk
  - https://github.com/agilexrobotics/piper_ros
"""

from __future__ import annotations

import argparse
import math
import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image

from deploy.client import Client as XiaomiModelClient

try:
    from piper_sdk import C_PiperInterface_V2 as PiperInterface
except Exception:
    from piper_sdk import C_PiperInterface as PiperInterface


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


class PiperSDKAdapter:
    """Translate model delta actions into PiPER SDK commands."""

    def __init__(
        self,
        can_port: str = "can0",
        move_speed_percent: int = 30,
        linear_step_m: float = 0.01,
        angular_step_rad: float = 0.12,
        gripper_open_m: float = 0.08,
        gripper_close_m: float = 0.0,
        gripper_effort: int = 1000,
        gripper_threshold: float = 0.0,
        safety: Optional[PiperSafety] = None,
    ) -> None:
        self.arm = PiperInterface(can_port)
        self.move_speed_percent = int(_clip(move_speed_percent, 1, 100))
        self.linear_step_m = float(linear_step_m)
        self.angular_step_rad = float(angular_step_rad)
        self.gripper_open_m = float(gripper_open_m)
        self.gripper_close_m = float(gripper_close_m)
        self.gripper_effort = int(_clip(gripper_effort, 0, 5000))
        self.gripper_threshold = float(gripper_threshold)
        self.safety = safety if safety is not None else PiperSafety()

        self._target_pose = np.zeros(
            6, dtype=np.float64
        )  # x,y,z,roll,pitch,yaw (m,rad)
        self._target_gripper_m = self.gripper_close_m

    def connect(self, enable_timeout_s: float = 5.0) -> None:
        self.arm.ConnectPort()
        time.sleep(0.2)
        self._enable_arm(enable_timeout_s)
        self._set_pose_mode()
        self.arm.GripperCtrl(0, self.gripper_effort, 0x01, 0)
        self._target_pose = self.read_end_pose_m_rad()
        self._target_gripper_m = self.read_gripper_m()

    def _enable_arm(self, timeout_s: float) -> None:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if hasattr(self.arm, "EnableArm"):
                self.arm.EnableArm(7, 0x02)
            elif hasattr(self.arm, "EnablePiper"):
                self.arm.EnablePiper()

            enabled = None
            if hasattr(self.arm, "GetArmEnableStatus"):
                try:
                    enabled = self.arm.GetArmEnableStatus()
                except Exception:
                    enabled = None

            if enabled is None:
                time.sleep(0.25)
                return

            if len(enabled) >= 6 and all(bool(x) for x in enabled[:6]):
                return
            time.sleep(0.05)

        raise RuntimeError(
            "PiPER enable timeout. Check CAN wiring/power and try again."
        )

    def _set_pose_mode(self) -> None:
        if hasattr(self.arm, "ModeCtrl"):
            self.arm.ModeCtrl(0x01, 0x00, self.move_speed_percent, 0x00)
            return
        if hasattr(self.arm, "MotionCtrl_2"):
            self.arm.MotionCtrl_2(0x01, 0x00, self.move_speed_percent, 0x00)
            return
        raise RuntimeError("No supported mode control API found in piper_sdk.")

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

    def build_calvin_state32(self) -> np.ndarray:
        pose = self.read_end_pose_m_rad()
        gripper = self.read_gripper_m()
        state7 = np.array(
            [pose[0], pose[1], pose[2], pose[3], pose[4], pose[5], gripper],
            dtype=np.float32,
        )
        return np.concatenate([state7, np.zeros(25, dtype=np.float32)], axis=0)

    def apply_delta_action(self, action7: np.ndarray) -> None:
        a = np.asarray(action7, dtype=np.float64).reshape(-1)
        if a.size < 7:
            raise ValueError(f"Expected at least 7 values, got {a.size}")

        delta_xyz = np.clip(a[:3], -1.0, 1.0) * self.linear_step_m
        delta_rpy = np.clip(a[3:6], -1.0, 1.0) * self.angular_step_rad

        self._target_pose[:3] = self._target_pose[:3] + delta_xyz
        self._target_pose[3:] = self._target_pose[3:] + delta_rpy

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

        # CALVIN eval binarizes gripper sign (>0 open, <=0 close).
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

        self._send_pose_and_gripper(self._target_pose, self._target_gripper_m)

    def _send_pose_and_gripper(self, pose6_m_rad: np.ndarray, gripper_m: float) -> None:
        self._set_pose_mode()

        x = int(round(float(pose6_m_rad[0]) * 1_000_000.0))
        y = int(round(float(pose6_m_rad[1]) * 1_000_000.0))
        z = int(round(float(pose6_m_rad[2]) * 1_000_000.0))
        rx = int(round(math.degrees(float(pose6_m_rad[3])) * 1000.0))
        ry = int(round(math.degrees(float(pose6_m_rad[4])) * 1000.0))
        rz = int(round(math.degrees(float(pose6_m_rad[5])) * 1000.0))

        self.arm.EndPoseCtrl(x, y, z, rx, ry, rz)

        gripper_val = int(round(abs(float(gripper_m)) * 1_000_000.0))
        self.arm.GripperCtrl(gripper_val, self.gripper_effort, 0x01, 0)

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


class XiaomiPiperController:
    """Model-server -> PiPER adapter bridge."""

    def __init__(
        self,
        model_host: str,
        model_port: int,
        piper_adapter: PiperSDKAdapter,
        task_id: str = "calvin_abcd_orig",
    ) -> None:
        self.model = XiaomiModelClient(host=model_host, port=model_port)
        self.piper = piper_adapter
        self.task_id = task_id

    def infer_action_chunk(
        self, base_img: Image.Image, wrist_img: Image.Image, instruction: str
    ) -> np.ndarray:
        base_img = _center_crop_keep_size(base_img.convert("RGB"), crop_ratio=0.95)
        wrist_img = _center_crop_keep_size(wrist_img.convert("RGB"), crop_ratio=0.95)

        model_inputs = {
            "task_id": self.task_id,
            "state": self.piper.build_calvin_state32(),
            "base": base_img,
            "wrist_left": wrist_img,
            "language": _normalize_instruction(instruction),
            "seed": int(time.time() * 1000) & 0xFFFFFFFF,
        }

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
        chunk = self.infer_action_chunk(base_img, wrist_img, instruction)
        # Follow eval_calvin behavior for gripper sign.
        chunk[:, 6] = np.where(chunk[:, 6] > 0.0, 1.0, -1.0)
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
    p.add_argument("--task-id", default="calvin_abcd_orig")
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
    p.add_argument("--ros-camera-mode", action="store_true", help="Use ROS image topics instead of OpenCV device indices")
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

    adapter = PiperSDKAdapter(can_port=args.can_port)
    adapter.connect()

    bridge = XiaomiPiperController(
        model_host=args.model_host,
        model_port=args.model_port,
        piper_adapter=adapter,
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
            args.wrist_cam_index if args.wrist_cam_index is not None else args.base_cam_index
        )
        if args.wrist_cam_index is None:
            print(
                "[WARN] --wrist-cam-index not provided. Reusing base camera as wrist view."
            )

        cap_base = _open_camera(
            args.base_cam_index, args.camera_width, args.camera_height, args.camera_fps
        )
        cap_wrist = _open_camera(
            wrist_index, args.camera_width, args.camera_height, args.camera_fps
        )
        try:
            while args.max_loops <= 0 or loops < args.max_loops:
                ok_base, frame_base = cap_base.read()
                ok_wrist, frame_wrist = cap_wrist.read()
                if not ok_base or not ok_wrist:
                    time.sleep(0.01)
                    continue

                base_img = _bgr_frame_to_pil(frame_base)
                wrist_img = _bgr_frame_to_pil(frame_wrist)

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
            cap_base.release()
            cap_wrist.release()

    finally:
        bridge.close()


if __name__ == "__main__":
    main()
