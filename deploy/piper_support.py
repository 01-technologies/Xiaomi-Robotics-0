from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional


CTRL_MODE_NAMES: dict[int, str] = {
    0x00: "standby",
    0x01: "can_command",
    0x02: "teach",
    0x03: "ethernet",
    0x04: "wifi",
    0x07: "offline_trajectory",
}

MOVE_MODE_NAMES: dict[int, str] = {
    0x00: "move_p",
    0x01: "move_j",
    0x02: "move_l",
    0x03: "move_c",
    0x04: "move_m",
    0x05: "move_cpv",
}

ARM_STATUS_NAMES: dict[int, str] = {
    0x00: "normal",
    0x01: "emergency_stop",
    0x02: "no_solution",
    0x03: "singularity",
    0x04: "target_angle_exceeds_limit",
    0x05: "abnormal_joint_communication",
    0x06: "joint_brake_not_open",
    0x07: "collision_protection",
    0x08: "overspeed_during_drag_teach",
    0x09: "abnormal_joint_condition",
    0x0A: "other_abnormality",
    0x0B: "teaching_record",
    0x0C: "teaching_execute",
    0x0D: "teaching_pause",
    0x0E: "main_control_overtemperature",
    0x0F: "release_resistor_overtemperature",
}

MOTION_STATUS_NAMES: dict[int, str] = {
    0x00: "arrived",
    0x01: "moving",
}


@dataclass(frozen=True)
class PiperDiagnosticReport:
    can_name: Optional[str] = None
    connect_status: Optional[bool] = None
    reader_ok: Optional[bool] = None
    can_fps: Optional[float] = None
    firmware_version: Optional[str] = None
    sdk_version: Optional[str] = None
    protocol_version: Optional[str] = None
    ctrl_mode: Optional[int] = None
    commanded_ctrl_mode: Optional[int] = None
    mode_feed: Optional[int] = None
    commanded_move_mode: Optional[int] = None
    arm_status: Optional[int] = None
    teach_status: Optional[int] = None
    motion_status: Optional[int] = None
    trajectory_num: Optional[int] = None
    enable_status: tuple[bool, ...] = ()
    joint_angle_limits: tuple[str, ...] = ()
    communication_faults: tuple[str, ...] = ()
    driver_disabled: tuple[str, ...] = ()
    driver_errors: tuple[str, ...] = ()
    collisions: tuple[str, ...] = ()
    stalls: tuple[str, ...] = ()
    voltage_low: tuple[str, ...] = ()
    motor_overheating: tuple[str, ...] = ()
    driver_overheating: tuple[str, ...] = ()
    driver_overcurrent: tuple[str, ...] = ()
    gripper_enabled: Optional[bool] = None
    gripper_homed: Optional[bool] = None
    gripper_driver_error: Optional[bool] = None
    gripper_angle_m: Optional[float] = None
    pose_m_rad: Optional[tuple[float, float, float, float, float, float]] = None
    notes: tuple[str, ...] = ()


def _format_hex(code: Optional[int]) -> str:
    return "unknown" if code is None else f"0x{code:02X}"


def _describe_code(code: Optional[int], mapping: dict[int, str]) -> str:
    if code is None:
        return "unknown"
    return mapping.get(code, f"unknown_{_format_hex(code)}")


def _to_optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _to_optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _to_optional_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    try:
        return bool(value)
    except Exception:
        return None


def _safe_attr(obj: Any, attr: str) -> Any:
    if obj is None:
        return None
    return getattr(obj, attr, None)


def _safe_call_method(
    obj: Any, method_name: str, *args: Any, **kwargs: Any
) -> tuple[Any, Optional[str]]:
    method = getattr(obj, method_name, None)
    if method is None:
        return None, f"{method_name} unavailable"
    try:
        return method(*args, **kwargs), None
    except Exception as exc:
        return None, f"{method_name} failed: {exc}"


def _coerce_bool_tuple(value: Any) -> tuple[bool, ...]:
    if value is None:
        return ()
    try:
        return tuple(bool(x) for x in list(value))
    except Exception:
        return ()


def _get_joint_driver_msg(msg: Any, joint_index: int) -> Any:
    for name in (
        f"motor_{joint_index}",
        f"joint_{joint_index}",
        f"motor{joint_index}",
        f"joint{joint_index}",
        f"m{joint_index}",
    ):
        candidate = _safe_attr(msg, name)
        if candidate is not None:
            return candidate
    return None


def collect_piper_diagnostic_report(arm: Any) -> PiperDiagnosticReport:
    notes: list[str] = []

    can_name, note = _safe_call_method(arm, "GetCanName")
    if note is not None:
        notes.append(note)

    connect_status, note = _safe_call_method(arm, "get_connect_status")
    if note is not None:
        notes.append(note)

    reader_ok, note = _safe_call_method(arm, "isOk")
    if note is not None:
        notes.append(note)

    can_fps, note = _safe_call_method(arm, "GetCanFps")
    if note is not None:
        notes.append(note)

    firmware_version, note = _safe_call_method(arm, "GetPiperFirmwareVersion")
    if note is not None:
        notes.append(note)

    sdk_version, note = _safe_call_method(arm, "GetCurrentSDKVersion")
    if note is not None:
        notes.append(note)

    protocol_version, note = _safe_call_method(arm, "GetCurrentProtocolVersion")
    if note is not None:
        notes.append(note)

    arm_status_msg, note = _safe_call_method(arm, "GetArmStatus")
    if note is not None:
        notes.append(note)
    status = _safe_attr(arm_status_msg, "arm_status")

    ctrl_mode = _to_optional_int(_safe_attr(status, "ctrl_mode"))
    mode_feed = _to_optional_int(_safe_attr(status, "mode_feed"))
    arm_status = _to_optional_int(_safe_attr(status, "arm_status"))
    teach_status = _to_optional_int(_safe_attr(status, "teach_status"))
    motion_status = _to_optional_int(_safe_attr(status, "motion_status"))
    trajectory_num = _to_optional_int(_safe_attr(status, "trajectory_num"))

    joint_angle_limits: list[str] = []
    communication_faults: list[str] = []
    err_status = _safe_attr(status, "err_status")
    for joint_index in range(1, 7):
        if bool(_safe_attr(err_status, f"joint_{joint_index}_angle_limit")):
            joint_angle_limits.append(f"joint_{joint_index}")
        if bool(_safe_attr(err_status, f"communication_status_joint_{joint_index}")):
            communication_faults.append(f"joint_{joint_index}")

    mode_msg, note = _safe_call_method(arm, "GetArmModeCtrl")
    if note is not None:
        notes.append(note)
    mode_ctrl = _safe_attr(mode_msg, "ctrl_151")
    if mode_ctrl is None:
        mode_ctrl = mode_msg
    commanded_ctrl_mode = _to_optional_int(_safe_attr(mode_ctrl, "ctrl_mode"))
    commanded_move_mode = _to_optional_int(_safe_attr(mode_ctrl, "move_mode"))

    enable_status, note = _safe_call_method(arm, "GetArmEnableStatus")
    if note is not None:
        notes.append(note)
    enable_status_tuple = _coerce_bool_tuple(enable_status)

    low_spd_msg, note = _safe_call_method(arm, "GetArmLowSpdInfoMsgs")
    if low_spd_msg is None:
        alt_msg, alt_note = _safe_call_method(arm, "GetDriverStates")
        if alt_note is not None:
            notes.append(alt_note)
        low_spd_msg = alt_msg
    elif note is not None:
        notes.append(note)

    driver_disabled: list[str] = []
    driver_errors: list[str] = []
    collisions: list[str] = []
    stalls: list[str] = []
    voltage_low: list[str] = []
    motor_overheating: list[str] = []
    driver_overheating: list[str] = []
    driver_overcurrent: list[str] = []
    for joint_index in range(1, 7):
        driver_msg = _get_joint_driver_msg(low_spd_msg, joint_index)
        foc_status = _safe_attr(driver_msg, "foc_status")
        if foc_status is None:
            continue
        joint_name = f"joint_{joint_index}"
        if _to_optional_bool(_safe_attr(foc_status, "driver_enable_status")) is False:
            driver_disabled.append(joint_name)
        if bool(_safe_attr(foc_status, "driver_error_status")):
            driver_errors.append(joint_name)
        if bool(_safe_attr(foc_status, "collision_status")):
            collisions.append(joint_name)
        if bool(_safe_attr(foc_status, "stall_status")):
            stalls.append(joint_name)
        if bool(_safe_attr(foc_status, "voltage_too_low")):
            voltage_low.append(joint_name)
        if bool(_safe_attr(foc_status, "motor_overheating")):
            motor_overheating.append(joint_name)
        if bool(_safe_attr(foc_status, "driver_overheating")):
            driver_overheating.append(joint_name)
        if bool(_safe_attr(foc_status, "driver_overcurrent")):
            driver_overcurrent.append(joint_name)

    gripper_msg, note = _safe_call_method(arm, "GetArmGripperMsgs")
    if note is not None:
        notes.append(note)
    gripper_state = _safe_attr(gripper_msg, "gripper_state")
    gripper_foc = _safe_attr(gripper_state, "foc_status")
    gripper_angle_raw = _to_optional_float(_safe_attr(gripper_state, "grippers_angle"))
    gripper_angle_m = None
    if gripper_angle_raw is not None:
        gripper_angle_m = gripper_angle_raw / 1_000_000.0

    pose_msg, note = _safe_call_method(arm, "GetArmEndPoseMsgs")
    if note is not None:
        notes.append(note)
    end_pose = _safe_attr(pose_msg, "end_pose")
    pose_m_rad = None
    x = _to_optional_float(_safe_attr(end_pose, "X_axis"))
    y = _to_optional_float(_safe_attr(end_pose, "Y_axis"))
    z = _to_optional_float(_safe_attr(end_pose, "Z_axis"))
    rx = _to_optional_float(_safe_attr(end_pose, "RX_axis"))
    ry = _to_optional_float(_safe_attr(end_pose, "RY_axis"))
    rz = _to_optional_float(_safe_attr(end_pose, "RZ_axis"))
    if all(value is not None for value in (x, y, z, rx, ry, rz)):
        assert x is not None
        assert y is not None
        assert z is not None
        assert rx is not None
        assert ry is not None
        assert rz is not None
        pose_m_rad = (
            float(x) / 1_000_000.0,
            float(y) / 1_000_000.0,
            float(z) / 1_000_000.0,
            math.radians(float(rx) / 1000.0),
            math.radians(float(ry) / 1000.0),
            math.radians(float(rz) / 1000.0),
        )

    return PiperDiagnosticReport(
        can_name=str(can_name) if can_name is not None else None,
        connect_status=_to_optional_bool(connect_status),
        reader_ok=_to_optional_bool(reader_ok),
        can_fps=_to_optional_float(can_fps),
        firmware_version=str(firmware_version)
        if firmware_version is not None
        else None,
        sdk_version=str(sdk_version) if sdk_version is not None else None,
        protocol_version=str(protocol_version)
        if protocol_version is not None
        else None,
        ctrl_mode=ctrl_mode,
        commanded_ctrl_mode=commanded_ctrl_mode,
        mode_feed=mode_feed,
        commanded_move_mode=commanded_move_mode,
        arm_status=arm_status,
        teach_status=teach_status,
        motion_status=motion_status,
        trajectory_num=trajectory_num,
        enable_status=enable_status_tuple,
        joint_angle_limits=tuple(joint_angle_limits),
        communication_faults=tuple(communication_faults),
        driver_disabled=tuple(driver_disabled),
        driver_errors=tuple(driver_errors),
        collisions=tuple(collisions),
        stalls=tuple(stalls),
        voltage_low=tuple(voltage_low),
        motor_overheating=tuple(motor_overheating),
        driver_overheating=tuple(driver_overheating),
        driver_overcurrent=tuple(driver_overcurrent),
        gripper_enabled=_to_optional_bool(
            _safe_attr(gripper_foc, "driver_enable_status")
        ),
        gripper_homed=_to_optional_bool(_safe_attr(gripper_foc, "homing_status")),
        gripper_driver_error=_to_optional_bool(
            _safe_attr(gripper_foc, "driver_error_status")
        ),
        gripper_angle_m=gripper_angle_m,
        pose_m_rad=pose_m_rad,
        notes=tuple(notes),
    )


def summarize_motion_blockers(report: PiperDiagnosticReport) -> tuple[str, ...]:
    blockers: list[str] = []

    actual_ctrl_mode = report.ctrl_mode
    if actual_ctrl_mode is not None and actual_ctrl_mode != 0x01:
        blockers.append(
            f"arm control mode is {_describe_code(actual_ctrl_mode, CTRL_MODE_NAMES)} ({_format_hex(actual_ctrl_mode)})"
        )
    elif report.commanded_ctrl_mode is not None and report.commanded_ctrl_mode != 0x01:
        blockers.append(
            f"commanded control mode is {_describe_code(report.commanded_ctrl_mode, CTRL_MODE_NAMES)} ({_format_hex(report.commanded_ctrl_mode)})"
        )

    if report.arm_status is not None and report.arm_status != 0x00:
        blockers.append(
            f"arm status is {_describe_code(report.arm_status, ARM_STATUS_NAMES)} ({_format_hex(report.arm_status)})"
        )

    if report.enable_status and not all(report.enable_status[:6]):
        disabled = [
            f"joint_{idx + 1}"
            for idx, enabled in enumerate(report.enable_status[:6])
            if not enabled
        ]
        blockers.append(f"disabled arm joints: {', '.join(disabled)}")

    if report.joint_angle_limits:
        blockers.append(
            f"joint angle limit flags: {', '.join(report.joint_angle_limits)}"
        )
    if report.communication_faults:
        blockers.append(
            f"joint communication faults: {', '.join(report.communication_faults)}"
        )
    if report.driver_disabled:
        blockers.append(f"driver not enabled: {', '.join(report.driver_disabled)}")
    if report.driver_errors:
        blockers.append(f"driver error flags: {', '.join(report.driver_errors)}")
    if report.collisions:
        blockers.append(f"collision flags: {', '.join(report.collisions)}")
    if report.stalls:
        blockers.append(f"stall flags: {', '.join(report.stalls)}")
    if report.voltage_low:
        blockers.append(f"low-voltage flags: {', '.join(report.voltage_low)}")
    if report.motor_overheating:
        blockers.append(f"motor overheat flags: {', '.join(report.motor_overheating)}")
    if report.driver_overheating:
        blockers.append(
            f"driver overheat flags: {', '.join(report.driver_overheating)}"
        )
    if report.driver_overcurrent:
        blockers.append(
            f"driver overcurrent flags: {', '.join(report.driver_overcurrent)}"
        )

    return tuple(blockers)


def suggest_piper_fixes(report: PiperDiagnosticReport) -> tuple[str, ...]:
    suggestions: list[str] = []

    if report.ctrl_mode == 0x02:
        suggestions.append(
            "Exit teach mode first, then switch the arm back to CAN control mode."
        )
    elif report.ctrl_mode is not None and report.ctrl_mode != 0x01:
        suggestions.append(
            "Switch the arm into CAN control mode before sending motion commands."
        )

    if report.arm_status == 0x01:
        suggestions.append(
            "Resume the emergency stop state with EmergencyStop(0x02) before retrying motion."
        )
    if report.arm_status in {0x02, 0x03, 0x04} or report.joint_angle_limits:
        suggestions.append(
            "Move back to a safer pose or smaller delta; the current pose/command looks outside IK or angle limits."
        )
    if report.arm_status == 0x06:
        suggestions.append(
            "The arm reports a brake issue; re-enable the joints and re-home/reset if needed."
        )
    if report.arm_status == 0x07 or report.collisions or report.stalls:
        suggestions.append(
            "Clear the collision/stall condition, then re-enable the arm from a safe zero pose."
        )
    if report.communication_faults:
        suggestions.append(
            "Check CAN power/wiring and confirm the CAN interface is up at 1000000 bitrate."
        )
    if report.enable_status and not all(report.enable_status[:6]):
        suggestions.append("Enable all six arm joints with EnableArm(7, 0x02).")
    if report.driver_disabled:
        suggestions.append(
            "One or more joint drivers are disabled; re-enable the arm and verify the teach button is off."
        )
    if (
        report.driver_errors
        or report.voltage_low
        or report.motor_overheating
        or report.driver_overheating
    ):
        suggestions.append(
            "Inspect joint driver health and power before retrying motion."
        )
    if not report.firmware_version:
        suggestions.append(
            "Verify the PiPER firmware version and SDK version; the V2 interface expects firmware V1.5-2 or newer."
        )

    if not suggestions:
        suggestions.append(
            "No obvious blocker is set in the status flags; next check zero-point / DH-offset calibration and test a very small Cartesian move."
        )
    return tuple(suggestions)


def format_piper_diagnostic_report(report: PiperDiagnosticReport) -> str:
    lines: list[str] = []

    blockers = summarize_motion_blockers(report)
    if blockers:
        lines.append(f"Blockers: {'; '.join(blockers)}")
    else:
        lines.append("Blockers: none detected in current status flags")

    lines.append(
        "Control: "
        f"actual={_describe_code(report.ctrl_mode, CTRL_MODE_NAMES)}({_format_hex(report.ctrl_mode)}), "
        f"commanded={_describe_code(report.commanded_ctrl_mode, CTRL_MODE_NAMES)}({_format_hex(report.commanded_ctrl_mode)}), "
        f"feedback_move={_describe_code(report.mode_feed, MOVE_MODE_NAMES)}({_format_hex(report.mode_feed)}), "
        f"commanded_move={_describe_code(report.commanded_move_mode, MOVE_MODE_NAMES)}({_format_hex(report.commanded_move_mode)}), "
        f"arm_status={_describe_code(report.arm_status, ARM_STATUS_NAMES)}({_format_hex(report.arm_status)}), "
        f"motion={_describe_code(report.motion_status, MOTION_STATUS_NAMES)}({_format_hex(report.motion_status)})"
    )

    if report.enable_status:
        enabled = " ".join(
            f"J{idx + 1}={'on' if flag else 'off'}"
            for idx, flag in enumerate(report.enable_status[:6])
        )
        lines.append(f"Enabled: {enabled}")

    if report.pose_m_rad is not None:
        x, y, z, rx, ry, rz = report.pose_m_rad
        lines.append(
            "Pose: "
            f"x={x:.3f}m y={y:.3f}m z={z:.3f}m "
            f"roll={math.degrees(rx):.1f}deg pitch={math.degrees(ry):.1f}deg yaw={math.degrees(rz):.1f}deg"
        )

    if report.gripper_angle_m is not None or report.gripper_enabled is not None:
        lines.append(
            "Gripper: "
            f"angle={report.gripper_angle_m if report.gripper_angle_m is not None else 'unknown'}m, "
            f"enabled={report.gripper_enabled}, homed={report.gripper_homed}, error={report.gripper_driver_error}"
        )

    lines.append(
        "SDK: "
        f"firmware={report.firmware_version}, sdk={report.sdk_version}, protocol={report.protocol_version}, "
        f"can={report.can_name}, connected={report.connect_status}, reader_ok={report.reader_ok}, can_fps={report.can_fps}"
    )

    if report.notes:
        lines.append(f"Notes: {'; '.join(report.notes)}")

    return "\n".join(lines)
