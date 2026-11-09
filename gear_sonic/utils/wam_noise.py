"""One sensor-noise definition for WAM collection, imagination and evaluation."""

import math

import torch


# SuperMimic OBS_NOISE, expressed as uniform half-widths in Sonic term order.
WAM_OBSERVATION_NOISE = {
    "gravity_dir": 0.05,
    "base_ang_vel": 0.2,
    "joint_pos": 0.01,
    "joint_vel": 0.5,
    "actions": 0.0,
}


def wam_observation_noise(settings=None):
    """Return a validated copy, including optional WAM-only overrides."""
    values = WAM_OBSERVATION_NOISE.copy()
    overrides = (settings or {}).get("observation_noise", {}) or {}
    if set(overrides) - set(values):
        raise ValueError("WAM observation_noise contains unknown proprioceptive terms")
    for name, value in overrides.items():
        value = float(value)
        if not math.isfinite(value) or value < 0:
            raise ValueError("WAM observation noise half-widths must be finite and nonnegative")
        values[name] = value
    return values


_CURRENT_ORIENTATION_TERMS = frozenset({
    "motion_anchor_ori_b", "motion_anchor_ori_b_nonflat", "smpl_root_ori_b",
})
_FUTURE_ORIENTATION_TERMS = {
    "motion_anchor_ori_b_mf_nonflat": False,
    "motion_anchor_ori_b_mf": True,
    "smpl_root_ori_b_multi_future": False,
    "smpl_root_ori_b_multi_future_flatten": True,
}


def supermimic_goal_noise_configs(term_names):
    """Return replacement tokenizer noise configs for the SuperMimic profile.

    Call only for physical WAM with ``domain_randomization=supermimic``. Assign
    every returned value, including ``None``, to the corresponding term's
    ``noise``. This removes inherited SONIC noise on all other goals. PPO and
    legacy WAM configs must not call this profile helper.

    The custom factory installs an actual callable in Isaac Lab's noise cfg;
    a plain ``func`` import string would not be resolved by custom_instantiate.
    No Isaac Lab dependency is imported until that config is instantiated.
    """
    configs = {}
    for name in term_names:
        if name in _FUTURE_ORIENTATION_TERMS:
            configs[name] = {
                "_target_": "gear_sonic.utils.wam_noise.current_frame_orientation_noise_cfg",
                "n_min": -0.05,
                "n_max": 0.05,
                "flattened": _FUTURE_ORIENTATION_TERMS[name],
            }
        elif name in _CURRENT_ORIENTATION_TERMS:
            configs[name] = {
                "_target_": "isaaclab.utils.noise.AdditiveUniformNoiseCfg",
                "n_min": -0.05,
                "n_max": 0.05,
            }
        else:
            configs[name] = None
    return configs


def _uniform_bounds(cfg):
    low, high = float(cfg.n_min), float(cfg.n_max)
    if not math.isfinite(low) or not math.isfinite(high) or low > high:
        raise ValueError("WAM reference noise needs finite ordered uniform bounds")
    if getattr(cfg, "operation", "add") != "add":
        raise ValueError("WAM reference noise requires additive uniform tokenizer noise")
    return low, high


def current_frame_orientation_noise(data, cfg):
    """Add noise only to frame zero of ``[..., reference_frames, 6]``.

    Leading dimensions are independent samples, including any imagination
    rollout dimension. Future references and the input tensor stay unchanged.
    """
    if data.ndim < 3 or data.shape[-1] != 6 or data.shape[-2] < 1:
        raise ValueError("Current-frame orientation noise expects [B,...,T,6] with T >= 1")
    if not data.is_floating_point():
        raise ValueError("Orientation noise requires floating-point observations")
    low, high = _uniform_bounds(cfg)
    result = data.clone(memory_format=torch.contiguous_format)
    current = result[..., 0, :]
    current.add_(low + torch.rand_like(current) * (high - low))
    return result


def current_frame_orientation_noise_flat(data, cfg):
    """Frame-zero noise for flattened ``[..., reference_frames * 6]`` terms."""
    if data.ndim < 2 or data.shape[-1] < 6 or data.shape[-1] % 6:
        raise ValueError("Flat orientation noise expects [B,...,T*6] with T >= 1")
    frames = data.reshape(*data.shape[:-1], -1, 6)
    return current_frame_orientation_noise(frames, cfg).reshape_as(data)


def current_frame_orientation_noise_cfg(n_min=-0.05, n_max=0.05, flattened=False):
    """Construct Isaac Lab's config lazily, keeping its noise func callable."""
    from isaaclab.utils.noise import AdditiveUniformNoiseCfg

    function = current_frame_orientation_noise_flat if flattened else current_frame_orientation_noise
    cfg = AdditiveUniformNoiseCfg(n_min=n_min, n_max=n_max, func=function)
    _uniform_bounds(cfg)
    return cfg


def reference_noise_spec(cfg):
    """Describe supported real noise without serializing an Isaac Lab object.

    Legacy uniform ranges retain their tuple representation in old replay
    checkpoints. The current-frame extension uses a plain, versioned-by-kind
    dictionary, so restored reference banks do not need the simulator runtime.
    """
    function = getattr(cfg, "func", None)
    for candidate, flattened in (
        (current_frame_orientation_noise, False),
        (current_frame_orientation_noise_flat, True),
    ):
        paths = {f"{candidate.__module__}:{candidate.__name__}",
                 f"{candidate.__module__}.{candidate.__name__}"}
        if function is candidate or (isinstance(function, str) and function in paths):
            low, high = _uniform_bounds(cfg)
            return {"kind": "current_reference_orientation", "n_min": low,
                    "n_max": high, "flattened": flattened}
    if not hasattr(cfg, "n_min") or not hasattr(cfg, "n_max"):
        raise ValueError("WAM reference noise requires additive uniform tokenizer noise")
    # Only known simulator uniform functions can share the legacy replay path.
    # A custom function with uniform-looking fields may have different semantics.
    if function is not None:
        module = getattr(function, "__module__", "")
        name = getattr(function, "__name__", "")
        if isinstance(function, str):
            module, _, name = function.replace(":", ".").rpartition(".")
        if not module.startswith("isaaclab.utils.noise") or name not in {
            "uniform_noise", "additive_uniform_noise",
        }:
            raise ValueError(f"Unsupported WAM reference noise function: {function}")
    return _uniform_bounds(cfg)


def apply_reference_noise(data, specification):
    """Apply a saved noise description with the real observation's shape."""
    if isinstance(specification, dict):
        if specification.get("kind") != "current_reference_orientation":
            raise ValueError("Unknown WAM reference noise description")
        from types import SimpleNamespace

        cfg = SimpleNamespace(n_min=specification["n_min"], n_max=specification["n_max"])
        function = (current_frame_orientation_noise_flat if specification["flattened"]
                    else current_frame_orientation_noise)
        return function(data, cfg)
    low, high = specification
    return data + low + torch.rand_like(data) * (high - low)
