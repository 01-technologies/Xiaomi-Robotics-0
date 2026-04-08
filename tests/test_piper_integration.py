"""Hardware-free regression tests for the PiPER adapter integration.

Tests cover:
  1. CALVIN gripper binary threshold behavior
  2. LIBERO gripper continuous & monotonic mapping
  3. Backend selection with SDK-only options
  4. Unsupported-option handling for piper_control
  5. Server import (sys.exit path)
  6. State-layout conventions (build_calvin_state32 / build_libero_state32)
"""

from __future__ import annotations

import math
import types
from typing import Any
from unittest import mock

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers: lightweight mock adapter for hardware-free tests
# ---------------------------------------------------------------------------


class _FakeArm:
    """Minimal stand-in for a PiPER SDK/piper_control arm object."""

    def __init__(self) -> None:
        self._pose = [0.2, 0.0, 0.15, 0.0, 0.0, 0.0]  # x,y,z,rx,ry,rz in m/rad
        self._gripper_m = 0.04

    # --- SDK-style feedback ---
    def GetArmEndPoseMsgs(self) -> Any:
        ep = types.SimpleNamespace(
            X_axis=int(self._pose[0] * 1_000_000),
            Y_axis=int(self._pose[1] * 1_000_000),
            Z_axis=int(self._pose[2] * 1_000_000),
            RX_axis=int(math.degrees(self._pose[3]) * 1000),
            RY_axis=int(math.degrees(self._pose[4]) * 1000),
            RZ_axis=int(math.degrees(self._pose[5]) * 1000),
        )
        return types.SimpleNamespace(end_pose=ep)

    def GetArmGripperMsgs(self) -> Any:
        gs = types.SimpleNamespace(
            grippers_angle=int(self._gripper_m * 1_000_000),
            foc_status=types.SimpleNamespace(
                driver_enable_status=True,
                homing_status=True,
                driver_error_status=False,
            ),
        )
        return types.SimpleNamespace(gripper_state=gs)

    def GetArmStatus(self) -> Any:
        return types.SimpleNamespace(
            arm_status=types.SimpleNamespace(
                ctrl_mode=0x01,
                mode_feed=0x00,
                arm_status=0x00,
                teach_status=0x00,
                motion_status=0x00,
                trajectory_num=0,
                err_status=types.SimpleNamespace(
                    **{f"joint_{i}_angle_limit": False for i in range(1, 7)},
                    **{f"communication_status_joint_{i}": False for i in range(1, 7)},
                ),
            )
        )

    def GetArmEnableStatus(self) -> list[bool]:
        return [True] * 7

    def GetArmLowSpdInfoMsgs(self) -> None:
        return None

    def GetArmModeCtrl(self) -> Any:
        return types.SimpleNamespace(
            ctrl_151=types.SimpleNamespace(ctrl_mode=0x01, move_mode=0x00),
        )

    def ConnectPort(self) -> None:
        pass

    def EnableArm(self, joint_index: int, mode: int = 0) -> None:
        pass

    def ModeCtrl(self, *args: Any) -> None:
        pass

    def EndPoseCtrl(self, *args: Any) -> None:
        pass

    def GripperCtrl(self, *args: Any) -> None:
        pass


def _make_sdk_adapter(
    gripper_mode: str = "binary",
    gripper_open_m: float = 0.08,
    gripper_close_m: float = 0.0,
    gripper_threshold: float = 0.0,
) -> Any:
    """Build a PiperSDKAdapter with a fake arm (no CAN bus needed)."""
    from deploy.piper_adapter import PiperSDKAdapter

    with mock.patch(
        "deploy.piper_adapter.build_piper_sdk_interface", return_value=_FakeArm()
    ):
        adapter = PiperSDKAdapter(
            can_port="can_fake",
            gripper_open_m=gripper_open_m,
            gripper_close_m=gripper_close_m,
            gripper_threshold=gripper_threshold,
            gripper_mode=gripper_mode,
        )
    adapter._sync_targets_from_feedback()
    return adapter


# ===========================================================================
# 1. CALVIN gripper regression: binary threshold
# ===========================================================================


class TestCalvinGripperBinary:
    """Prove that binary threshold behavior is preserved for CALVIN."""

    def test_positive_input_opens(self) -> None:
        adapter = _make_sdk_adapter(gripper_mode="binary")
        action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5])
        adapter.apply_delta_action(action)
        assert adapter._target_gripper_m == pytest.approx(0.08)

    def test_negative_input_closes(self) -> None:
        adapter = _make_sdk_adapter(gripper_mode="binary")
        action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5])
        adapter.apply_delta_action(action)
        assert adapter._target_gripper_m == pytest.approx(0.0)

    def test_zero_input_closes(self) -> None:
        adapter = _make_sdk_adapter(gripper_mode="binary", gripper_threshold=0.0)
        action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        adapter.apply_delta_action(action)
        assert adapter._target_gripper_m == pytest.approx(0.0)

    def test_only_two_output_values(self) -> None:
        adapter = _make_sdk_adapter(gripper_mode="binary")
        outputs = set()
        for v in np.linspace(-1.0, 1.0, 21):
            action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, v])
            adapter.apply_delta_action(action)
            outputs.add(round(adapter._target_gripper_m, 6))
        assert outputs == {0.0, 0.08}


# ===========================================================================
# 2. LIBERO gripper regression: continuous & monotonic
# ===========================================================================


class TestLiberoGripperContinuous:
    """Prove LIBERO gripper is continuous, not collapsed to two values."""

    def test_representative_inputs_not_collapsed(self) -> None:
        adapter = _make_sdk_adapter(gripper_mode="continuous")
        inputs = [-1.0, -0.5, 0.0, 0.5, 1.0]
        outputs = []
        for v in inputs:
            action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, v])
            adapter.apply_delta_action(action)
            outputs.append(adapter._target_gripper_m)
        # Must have more than two unique values
        assert len(set(round(o, 6) for o in outputs)) > 2

    def test_monotonic_mapping(self) -> None:
        adapter = _make_sdk_adapter(gripper_mode="continuous")
        inputs = np.linspace(-1.0, 1.0, 51)
        outputs = []
        for v in inputs:
            action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, v])
            adapter.apply_delta_action(action)
            outputs.append(adapter._target_gripper_m)
        # Monotonically non-decreasing (open_m > close_m)
        for i in range(1, len(outputs)):
            assert outputs[i] >= outputs[i - 1] - 1e-9

    def test_endpoints(self) -> None:
        adapter = _make_sdk_adapter(
            gripper_mode="continuous",
            gripper_open_m=0.08,
            gripper_close_m=0.0,
        )
        # -1 should map to close
        action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
        adapter.apply_delta_action(action)
        assert adapter._target_gripper_m == pytest.approx(0.0)
        # +1 should map to open
        action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        adapter.apply_delta_action(action)
        assert adapter._target_gripper_m == pytest.approx(0.08)

    def test_midpoint(self) -> None:
        adapter = _make_sdk_adapter(
            gripper_mode="continuous",
            gripper_open_m=0.08,
            gripper_close_m=0.0,
        )
        action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        adapter.apply_delta_action(action)
        assert adapter._target_gripper_m == pytest.approx(0.04)

    def test_values_outside_range_are_clipped(self) -> None:
        adapter = _make_sdk_adapter(gripper_mode="continuous")
        action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0])
        adapter.apply_delta_action(action)
        assert adapter._target_gripper_m == pytest.approx(0.08)

        action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.0])
        adapter.apply_delta_action(action)
        assert adapter._target_gripper_m == pytest.approx(0.0)


# ===========================================================================
# 3. Backend-selection regression
# ===========================================================================


class TestBackendSelection:
    """Prove auto prefers piper_sdk when SDK-only options are requested."""

    def test_auto_default_prefers_piper_control_when_available(self) -> None:
        from deploy.piper_backend_support import resolve_piper_backend

        with mock.patch(
            "deploy.piper_backend_support.load_piper_control_modules",
            return_value=mock.MagicMock(),
        ):
            result = resolve_piper_backend("auto", sdk_only_options_requested=False)
        assert result == "piper_control"

    def test_auto_prefers_piper_sdk_with_sdk_only_options(self) -> None:
        from deploy.piper_backend_support import resolve_piper_backend

        # Even if piper_control is importable, SDK-only options should force piper_sdk
        with mock.patch(
            "deploy.piper_backend_support.load_piper_control_modules",
            return_value=mock.MagicMock(),
        ):
            result = resolve_piper_backend("auto", sdk_only_options_requested=True)
        assert result == "piper_sdk"

    def test_auto_falls_back_to_piper_sdk_when_control_unavailable(self) -> None:
        from deploy.piper_backend_support import resolve_piper_backend

        with mock.patch(
            "deploy.piper_backend_support.load_piper_control_modules",
            side_effect=ImportError("not installed"),
        ):
            result = resolve_piper_backend("auto", sdk_only_options_requested=False)
        assert result == "piper_sdk"

    def test_has_sdk_only_options_detects_non_defaults(self) -> None:
        from deploy.piper_backend_support import has_sdk_only_options

        assert has_sdk_only_options() is False
        assert has_sdk_only_options(judge_flag=False) is True
        assert has_sdk_only_options(dh_is_offset=1) is True
        assert has_sdk_only_options(sdk_joint_limit=True) is True
        assert has_sdk_only_options(sdk_gripper_limit=False) is True
        assert has_sdk_only_options(force_slave_mode=True) is True


# ===========================================================================
# 4. Unsupported-option regression for piper_control
# ===========================================================================


class TestPiperControlUnsupportedOptions:
    """Prove piper_control raises ValueError (not just warns) for SDK-only flags."""

    def _make_fake_piper_control_modules(self) -> mock.MagicMock:
        fake_robot = mock.MagicMock()
        fake_robot.gripper_angle_max = 0.08
        fake_robot.gripper_effort_max = 1.0
        fake_robot.piper = _FakeArm()
        fake_modules = mock.MagicMock()
        return fake_robot, fake_modules

    def test_judge_flag_false_raises(self) -> None:
        from deploy.piper_adapter import PiperControlAdapter

        fake_robot, fake_modules = self._make_fake_piper_control_modules()
        with mock.patch(
            "deploy.piper_adapter.build_piper_control_interface",
            return_value=(fake_robot, fake_modules),
        ):
            with pytest.raises(ValueError, match="judge_flag"):
                PiperControlAdapter(judge_flag=False)

    def test_dh_is_offset_raises(self) -> None:
        from deploy.piper_adapter import PiperControlAdapter

        fake_robot, fake_modules = self._make_fake_piper_control_modules()
        with mock.patch(
            "deploy.piper_adapter.build_piper_control_interface",
            return_value=(fake_robot, fake_modules),
        ):
            with pytest.raises(ValueError, match="dh_is_offset"):
                PiperControlAdapter(dh_is_offset=1)

    def test_force_slave_mode_raises(self) -> None:
        from deploy.piper_adapter import PiperControlAdapter

        fake_robot, fake_modules = self._make_fake_piper_control_modules()
        with mock.patch(
            "deploy.piper_adapter.build_piper_control_interface",
            return_value=(fake_robot, fake_modules),
        ):
            with pytest.raises(ValueError, match="force_slave_mode"):
                PiperControlAdapter(force_slave_mode=True)

    def test_no_error_when_defaults(self) -> None:
        from deploy.piper_adapter import PiperControlAdapter

        fake_robot, fake_modules = self._make_fake_piper_control_modules()
        with mock.patch(
            "deploy.piper_adapter.build_piper_control_interface",
            return_value=(fake_robot, fake_modules),
        ):
            adapter = PiperControlAdapter()
            assert adapter.backend_name == "piper_control"


# ===========================================================================
# 5. Server import regression
# ===========================================================================


class TestServerImport:
    """Cover the deploy/server.py sys.exit path."""

    def test_server_module_imports_sys(self) -> None:
        import deploy.server as server_mod

        assert hasattr(server_mod, "sys"), "deploy.server must import sys"

    def test_sys_exit_is_callable_in_server(self) -> None:
        import deploy.server as server_mod

        assert callable(server_mod.sys.exit)


# ===========================================================================
# 6. State-layout regression
# ===========================================================================


class TestStateLayouts:
    """Verify build_calvin_state32 and build_libero_state32 shape/content."""

    def _make_adapter(self) -> Any:
        return _make_sdk_adapter(gripper_mode="binary")

    def test_calvin_state32_shape(self) -> None:
        adapter = self._make_adapter()
        state = adapter.build_calvin_state32()
        assert state.shape == (32,)
        assert state.dtype == np.float32

    def test_calvin_state32_structure(self) -> None:
        adapter = self._make_adapter()
        state = adapter.build_calvin_state32()
        # First 7 values: [x, y, z, roll, pitch, yaw, gripper]
        # Remaining 25: zeros
        assert np.all(state[7:] == 0.0)
        # Non-trivial pose values (x=0.2, z=0.15 from fake arm)
        assert state[0] == pytest.approx(0.2, abs=0.01)
        assert state[2] == pytest.approx(0.15, abs=0.01)
        # Gripper value
        assert state[6] == pytest.approx(0.04, abs=0.01)

    def test_libero_state32_shape(self) -> None:
        adapter = self._make_adapter()
        state = adapter.build_libero_state32()
        assert state.shape == (32,)
        assert state.dtype == np.float32

    def test_libero_state32_structure(self) -> None:
        adapter = self._make_adapter()
        state = adapter.build_libero_state32()
        # First 3: position [x, y, z]
        assert state[0] == pytest.approx(0.2, abs=0.01)
        assert state[2] == pytest.approx(0.15, abs=0.01)
        # Dims 3-5: axis-angle rotation (from euler zero → near-zero axis-angle)
        assert np.allclose(state[3:6], 0.0, atol=0.01)
        # Dims 6-7: finger_qpos = 0.5 * gripper_width
        expected_finger = 0.5 * 0.04  # gripper = 0.04m
        assert state[6] == pytest.approx(expected_finger, abs=0.005)
        assert state[7] == pytest.approx(expected_finger, abs=0.005)
        # Remaining 24: zeros
        assert np.all(state[8:] == 0.0)


# ===========================================================================
# 7. build_piper_control_interface kwargs passthrough
# ===========================================================================


class TestPiperControlInterfaceKwargs:
    """Verify signature-based kwargs passthrough in build_piper_control_interface."""

    def test_extra_kwargs_passed_when_supported(self) -> None:
        from deploy.piper_backend_support import build_piper_control_interface

        # Create a fake PiperInterface class that accepts extra kwargs
        class FakePiperInterface:
            def __init__(self, can_port: str, speed: int = 10) -> None:
                self.can_port = can_port
                self.speed = speed

        fake_modules = mock.MagicMock()
        fake_modules.piper_interface.PiperInterface = FakePiperInterface

        with mock.patch(
            "deploy.piper_backend_support.load_piper_control_modules",
            return_value=fake_modules,
        ):
            robot, _ = build_piper_control_interface("can0", speed=42)
        assert robot.speed == 42

    def test_unsupported_kwargs_silently_ignored(self) -> None:
        from deploy.piper_backend_support import build_piper_control_interface

        class FakePiperInterface:
            def __init__(self, can_port: str) -> None:
                self.can_port = can_port

        fake_modules = mock.MagicMock()
        fake_modules.piper_interface.PiperInterface = FakePiperInterface

        with mock.patch(
            "deploy.piper_backend_support.load_piper_control_modules",
            return_value=fake_modules,
        ):
            robot, _ = build_piper_control_interface("can0", nonexistent_arg=99)
        assert robot.can_port == "can0"
