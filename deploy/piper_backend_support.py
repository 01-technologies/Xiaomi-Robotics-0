from __future__ import annotations

import importlib
import inspect
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional


PiperBackendName = Literal["piper_control", "piper_sdk"]
PIPER_BACKEND_CHOICES = ("auto", "piper_control", "piper_sdk")


@dataclass(frozen=True)
class PiperControlModules:
    piper_init: Any
    piper_interface: Any


def _normalize_piper_control_src(path_str: Optional[str]) -> Optional[Path]:
    if not path_str:
        return None

    candidate = Path(path_str).expanduser().resolve()
    if (candidate / "piper_control").is_dir():
        return candidate
    if (candidate / "src" / "piper_control").is_dir():
        return candidate / "src"
    return None


def _iter_piper_control_src_candidates(
    explicit_src: Optional[str] = None,
) -> tuple[Path, ...]:
    base_dir = Path(__file__).resolve().parents[2]
    candidates = (
        explicit_src,
        os.environ.get("PIPER_CONTROL_SRC"),
        str(base_dir / "piper_control"),
        str(base_dir / "piper_control" / "src"),
    )

    seen: set[Path] = set()
    normalized: list[Path] = []
    for candidate in candidates:
        resolved = _normalize_piper_control_src(candidate)
        if resolved is None or resolved in seen:
            continue
        seen.add(resolved)
        normalized.append(resolved)
    return tuple(normalized)


def load_piper_control_modules(
    piper_control_src: Optional[str] = None,
) -> PiperControlModules:
    def _import() -> PiperControlModules:
        return PiperControlModules(
            piper_init=importlib.import_module("piper_control.piper_init"),
            piper_interface=importlib.import_module("piper_control.piper_interface"),
        )

    try:
        return _import()
    except ModuleNotFoundError as exc:
        if exc.name not in {
            "piper_control",
            "piper_control.piper_init",
            "piper_control.piper_interface",
        }:
            raise

    for candidate in _iter_piper_control_src_candidates(piper_control_src):
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)
        try:
            return _import()
        except ModuleNotFoundError as exc:
            if exc.name not in {
                "piper_control",
                "piper_control.piper_init",
                "piper_control.piper_interface",
            }:
                raise

    searched = ", ".join(
        str(path) for path in _iter_piper_control_src_candidates(piper_control_src)
    )
    raise ModuleNotFoundError(
        "Unable to import piper_control. Install it or point PIPER_CONTROL_SRC/"
        "--piper-control-src to a checkout containing src/piper_control"
        + (f". Searched: {searched}" if searched else ".")
    )


def has_sdk_only_options(
    judge_flag: bool = True,
    dh_is_offset: Optional[int] = None,
    sdk_joint_limit: Optional[bool] = None,
    sdk_gripper_limit: Optional[bool] = None,
    force_slave_mode: bool = False,
) -> bool:
    """Return True when any SDK-only constructor knob is set to a non-default value."""
    return (
        not judge_flag
        or dh_is_offset is not None
        or sdk_joint_limit is not None
        or sdk_gripper_limit is not None
        or force_slave_mode
    )


def resolve_piper_backend(
    requested_backend: str,
    piper_control_src: Optional[str] = None,
    *,
    sdk_only_options_requested: bool = False,
) -> PiperBackendName:
    normalized = requested_backend.strip().lower()
    if normalized == "auto":
        if sdk_only_options_requested:
            return "piper_sdk"
        try:
            load_piper_control_modules(piper_control_src)
        except ImportError:
            return "piper_sdk"
        return "piper_control"
    if normalized == "piper_control":
        load_piper_control_modules(piper_control_src)
        return "piper_control"
    if normalized == "piper_sdk":
        return "piper_sdk"
    raise ValueError(
        f"Unsupported PiPER backend '{requested_backend}'. Expected one of {PIPER_BACKEND_CHOICES}."
    )


def _resolve_piper_sdk_interface_class() -> Any:
    try:
        from piper_sdk import C_PiperInterface_V2 as interface_cls
    except Exception:
        from piper_sdk import C_PiperInterface as interface_cls
    return interface_cls


def build_piper_sdk_interface(
    can_port: str,
    judge_flag: bool = True,
    dh_is_offset: Optional[int] = None,
    sdk_joint_limit: Optional[bool] = None,
    sdk_gripper_limit: Optional[bool] = None,
) -> Any:
    interface_cls = _resolve_piper_sdk_interface_class()
    kwargs: dict[str, Any] = {"judge_flag": judge_flag}
    if dh_is_offset is not None:
        kwargs["dh_is_offset"] = dh_is_offset
    if sdk_joint_limit is not None:
        kwargs["start_sdk_joint_limit"] = sdk_joint_limit
    if sdk_gripper_limit is not None:
        kwargs["start_sdk_gripper_limit"] = sdk_gripper_limit

    call_patterns: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    call_patterns.append(((can_port,), kwargs))
    call_patterns.append(((), {"can_name": can_port, **kwargs}))
    call_patterns.append(((can_port,), {"judge_flag": judge_flag}))
    call_patterns.append(((), {"can_name": can_port, "judge_flag": judge_flag}))
    call_patterns.append(((can_port,), {}))
    call_patterns.append(((), {"can_name": can_port}))

    last_error: Optional[Exception] = None
    for args, kw in call_patterns:
        try:
            return interface_cls(*args, **kw)
        except TypeError as exc:
            last_error = exc

    if last_error is not None:
        raise last_error
    raise RuntimeError("Failed to construct a PiPER SDK interface.")


def build_piper_control_interface(
    can_port: str,
    piper_control_src: Optional[str] = None,
    **extra_kwargs: Any,
) -> tuple[Any, PiperControlModules]:
    modules = load_piper_control_modules(piper_control_src)
    piper_interface = modules.piper_interface
    cls = piper_interface.PiperInterface

    # Determine which kwargs the installed PiperInterface accepts.
    try:
        sig = inspect.signature(cls.__init__)
        supported_params = set(sig.parameters.keys()) - {"self"}
    except (ValueError, TypeError):
        supported_params = {"can_port"}

    kwargs: dict[str, Any] = {"can_port": can_port}
    for key, value in extra_kwargs.items():
        if key in supported_params:
            kwargs[key] = value

    return cls(**kwargs), modules
