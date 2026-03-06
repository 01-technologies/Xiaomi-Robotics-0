#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from typing import Any

from deploy.piper_support import (
    PiperDiagnosticReport,
    PiperInterface,
    collect_piper_diagnostic_report,
    format_piper_diagnostic_report,
    suggest_piper_fixes,
)


def _build_arm(can_port: str, judge_flag: bool):
    try:
        return PiperInterface(can_port, judge_flag=judge_flag)
    except TypeError:
        return PiperInterface(can_port)


def _connect_arm(arm: Any) -> None:
    arm.ConnectPort()
    time.sleep(0.2)


def _resume_estop(arm: Any) -> bool:
    if not hasattr(arm, "EmergencyStop"):
        return False
    arm.EmergencyStop(0x02)
    time.sleep(0.1)
    return True


def _enable_arm(arm: Any, timeout_s: float) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if hasattr(arm, "EnableArm"):
            arm.EnableArm(7, 0x02)
        elif hasattr(arm, "EnablePiper"):
            arm.EnablePiper()
        else:
            return False

        if hasattr(arm, "GetArmEnableStatus"):
            try:
                enabled = arm.GetArmEnableStatus()
            except Exception:
                enabled = None
            if (
                enabled is not None
                and len(enabled) >= 6
                and all(bool(x) for x in enabled[:6])
            ):
                return True
        else:
            return True
        time.sleep(0.05)
    return False


def _set_pose_mode(arm: Any, move_speed_percent: int) -> bool:
    if hasattr(arm, "ModeCtrl"):
        arm.ModeCtrl(0x01, 0x00, move_speed_percent, 0x00)
        return True
    if hasattr(arm, "MotionCtrl_2"):
        arm.MotionCtrl_2(0x01, 0x00, move_speed_percent, 0x00)
        return True
    return False


def _read_pose_m_rad(arm: Any) -> tuple[float, float, float, float, float, float]:
    msg = arm.GetArmEndPoseMsgs()
    pose = msg.end_pose
    return (
        float(pose.X_axis) / 1_000_000.0,
        float(pose.Y_axis) / 1_000_000.0,
        float(pose.Z_axis) / 1_000_000.0,
        math.radians(float(pose.RX_axis) / 1000.0),
        math.radians(float(pose.RY_axis) / 1000.0),
        math.radians(float(pose.RZ_axis) / 1000.0),
    )


def _send_end_pose(
    arm: Any, pose_m_rad: tuple[float, float, float, float, float, float]
) -> None:
    x = int(round(pose_m_rad[0] * 1_000_000.0))
    y = int(round(pose_m_rad[1] * 1_000_000.0))
    z = int(round(pose_m_rad[2] * 1_000_000.0))
    rx = int(round(math.degrees(pose_m_rad[3]) * 1000.0))
    ry = int(round(math.degrees(pose_m_rad[4]) * 1000.0))
    rz = int(round(math.degrees(pose_m_rad[5]) * 1000.0))
    try:
        arm.EndPoseCtrl(x, y, z, rx, ry, rz)
    except Exception as exc:
        raise RuntimeError(
            "Failed to send PiPER EndPoseCtrl during diagnostic move.\n"
            f"{format_piper_diagnostic_report(collect_piper_diagnostic_report(arm))}"
        ) from exc


def _run_cartesian_test(
    arm: Any,
    axis: str,
    delta_mm: float,
    wait_s: float,
    move_speed_percent: int,
) -> dict[str, tuple[float, float, float, float, float, float]]:
    if not _set_pose_mode(arm, move_speed_percent):
        raise RuntimeError(
            "Unable to switch PiPER into pose mode before the motion test."
        )

    start = _read_pose_m_rad(arm)
    target = list(start)
    axis_index = {"x": 0, "y": 1, "z": 2}[axis]
    target[axis_index] += delta_mm / 1000.0
    target_pose: tuple[float, float, float, float, float, float] = (
        float(target[0]),
        float(target[1]),
        float(target[2]),
        float(target[3]),
        float(target[4]),
        float(target[5]),
    )
    _send_end_pose(arm, target_pose)
    time.sleep(wait_s)
    moved = _read_pose_m_rad(arm)
    _send_end_pose(arm, start)
    time.sleep(wait_s)
    returned = _read_pose_m_rad(arm)
    return {
        "start": start,
        "moved": moved,
        "returned": returned,
    }


def _print_report(report: PiperDiagnosticReport, title: str) -> None:
    print(title)
    print(format_piper_diagnostic_report(report))
    print("Suggested fixes:")
    for suggestion in suggest_piper_fixes(report):
        print(f"- {suggestion}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect PiPER control state and common motion blockers."
    )
    parser.add_argument("--can-port", default="can0")
    parser.add_argument(
        "--judge-flag",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass judge_flag to piper_sdk when supported. Use --no-judge-flag for some PCIe-to-CAN adapters.",
    )
    parser.add_argument(
        "--try-recover",
        action="store_true",
        help="Attempt a soft recovery: resume E-stop, enable all joints, and switch to CAN pose mode.",
    )
    parser.add_argument(
        "--move-speed-percent",
        type=int,
        default=30,
        help="Speed percent used for --try-recover and --test-cartesian-mm.",
    )
    parser.add_argument(
        "--enable-timeout",
        type=float,
        default=5.0,
        help="Timeout in seconds for soft-recovery joint enabling.",
    )
    parser.add_argument(
        "--test-cartesian-mm",
        type=float,
        default=0.0,
        help="Optional read/write test: move a small Cartesian delta, then return to the start pose.",
    )
    parser.add_argument(
        "--test-axis",
        choices=("x", "y", "z"),
        default="x",
        help="Axis used by --test-cartesian-mm.",
    )
    parser.add_argument(
        "--test-wait",
        type=float,
        default=0.5,
        help="Seconds to wait after each motion command in the optional Cartesian test.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the diagnostic report as JSON instead of human-readable text.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    arm = _build_arm(args.can_port, args.judge_flag)
    _connect_arm(arm)

    payload: dict[str, Any] = {}
    before = collect_piper_diagnostic_report(arm)
    if args.json:
        payload = {
            "before": asdict(before),
            "before_suggestions": list(suggest_piper_fixes(before)),
        }
    else:
        _print_report(before, "Current PiPER diagnostics")

    if args.try_recover:
        actions: list[str] = []
        if _resume_estop(arm):
            actions.append("resume emergency stop")
        if _enable_arm(arm, timeout_s=args.enable_timeout):
            actions.append("enable all arm joints")
        if _set_pose_mode(arm, args.move_speed_percent):
            actions.append("set CAN pose mode")

        after = collect_piper_diagnostic_report(arm)
        if args.json:
            payload["recovery_actions"] = actions
            payload["after_recovery"] = asdict(after)
            payload["after_recovery_suggestions"] = list(suggest_piper_fixes(after))
        else:
            print()
            print(f"Recovery actions: {', '.join(actions) if actions else 'none'}")
            _print_report(after, "Diagnostics after soft recovery")

    if abs(args.test_cartesian_mm) > 1e-9:
        result = _run_cartesian_test(
            arm,
            axis=args.test_axis,
            delta_mm=args.test_cartesian_mm,
            wait_s=args.test_wait,
            move_speed_percent=args.move_speed_percent,
        )
        if args.json:
            payload["cartesian_test"] = result
        else:
            start = result["start"]
            moved = result["moved"]
            returned = result["returned"]
            print()
            print(
                "Cartesian test: "
                f"axis={args.test_axis}, requested_delta_mm={args.test_cartesian_mm:.2f}, "
                f"observed_delta_mm={(moved[{'x': 0, 'y': 1, 'z': 2}[args.test_axis]] - start[{'x': 0, 'y': 1, 'z': 2}[args.test_axis]]) * 1000.0:.2f}, "
                f"return_error_mm={(returned[{'x': 0, 'y': 1, 'z': 2}[args.test_axis]] - start[{'x': 0, 'y': 1, 'z': 2}[args.test_axis]]) * 1000.0:.2f}"
            )

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
