import math
from contextlib import nullcontext
from functools import lru_cache
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from loguru import logger
from omegaconf import OmegaConf, open_dict

from gear_sonic.utils.wam_noise import wam_observation_noise


_WAM_OBSERVATION_GROUPS = {
    "wam": "proprio",
    "wam_state": "physical",
    "wam_reference": "physical",
    "wam_context": "physical",
}


@lru_cache(maxsize=1)
def _wam_observation_templates():
    """Resolve group defaults with Hydra, without instantiating Isaac Lab terms."""
    config_dir = Path(__file__).resolve().parents[1] / "config"
    # Training already owns Hydra's config directory; loaded checkpoints and
    # standalone tools may call the same helpers outside a Hydra application.
    scope = nullcontext() if GlobalHydra.instance().is_initialized() else initialize_config_dir(
        config_dir=str(config_dir), version_base="1.1",
    )
    with scope:
        templates = compose(config_name=None, overrides=[
            f"+manager_env/observations/{name}={selection}"
            for name, selection in _WAM_OBSERVATION_GROUPS.items()
        ])
    return OmegaConf.to_container(templates.manager_env.observations, resolve=False)


def _configure_wam_observation_groups(config):
    """Opt physical WAM into its extra groups; remove legacy copies from PPO.

    Group templates are merged before user or saved-checkpoint overrides, so
    repeated calls preserve explicit terms and contain no unresolved defaults.
    Only the four reserved WAM groups are removed from other trainers.
    """
    target = str(OmegaConf.select(config, "trainer._target_", default=""))
    is_wam = target.rsplit(".", 1)[-1] == "WAMTrainer"
    observations = OmegaConf.select(config, "manager_env.observations")
    if not is_wam:
        if observations is not None:
            with open_dict(observations):
                for name in _WAM_OBSERVATION_GROUPS:
                    observations.pop(name, None)
        return False
    if OmegaConf.select(config, "world_model.cfg.state_type", default="flat") != "physical":
        # Flat adapters can still explicitly request the legacy WAM groups.
        # Add only missing requested groups; their custom observation contract
        # must not be merged with physical defaults or sensor-noise settings.
        requested = set()
        for key in ("state_observation_keys", "context_observation_keys",
                    "reference_observation_keys", "goal_observation_keys", "observation_keys"):
            requested.update(OmegaConf.select(config, f"trainer.settings.{key}", default=[]) or [])
        requested.intersection_update(_WAM_OBSERVATION_GROUPS)
        if "wam_state" in requested:
            requested.add("wam")
        if requested:
            if observations is None:
                raise ValueError("WAM observation keys require manager_env.observations")
            defaults = _wam_observation_templates()
            with open_dict(observations):
                for name in _WAM_OBSERVATION_GROUPS:
                    if name in requested and observations.get(name) is None:
                        observations[name] = OmegaConf.create(defaults[name])
        return False
    terrain = OmegaConf.select(config, "trainer.settings.terrain_type")
    if terrain is not None and (not isinstance(terrain, str) or terrain not in {"plane", "trimesh"}):
        raise ValueError("WAM terrain_type must be null, 'plane', or 'trimesh'")
    if observations is None:
        raise ValueError("Physical WAM requires manager_env.observations")
    defaults = _wam_observation_templates()
    with open_dict(observations):
        for name in _WAM_OBSERVATION_GROUPS:
            overrides = observations.get(name)
            observations[name] = OmegaConf.merge(defaults[name], overrides or {})
    if terrain is not None:
        OmegaConf.update(config, "manager_env.config.terrain_type", terrain, force_add=True)
    return True


def configure_wam_context(config):
    """Enable detailed simulator context only for the physical WAM branch."""
    if not _configure_wam_observation_groups(config):
        return False
    enabled = bool(OmegaConf.select(config, "trainer.settings.complete_context", default=False))
    term = OmegaConf.select(config, "manager_env.observations.wam_context.xi")
    if term is None:
        raise ValueError("Complete physical WAM context requires the wam_context.xi observation")
    with open_dict(term):
        if term.get("params") is None:
            term.params = {}
        with open_dict(term.params):
            term.params.complete = enabled
    return True


def configure_wam_observations(config):
    """Use the same WAM sensor noise for policy evaluation and imagined frames.

    Collection builds the actor history from the single-frame ``wam`` group;
    evaluation uses the native ``policy`` history. Both groups therefore need
    identical per-term corruption. PPO drops only the reserved WAM groups;
    flat WAM configurations keep their existing observation contract.
    """
    if not _configure_wam_observation_groups(config):
        return False
    widths = wam_observation_noise(OmegaConf.select(config, "trainer.settings", default={}))
    for name in ("policy", "wam"):
        group = OmegaConf.select(config, f"manager_env.observations.{name}")
        if group is None:
            raise ValueError(f"Physical WAM requires the {name!r} observation group")
        with open_dict(group):
            group.enable_corruption = any(width > 0 for width in widths.values())
            for term_name, width in widths.items():
                if term_name in group and group[term_name] is not None:
                    with open_dict(group[term_name]):
                        group[term_name].noise = {
                            "_target_": "isaaclab.utils.noise.AdditiveUniformNoiseCfg",
                            "n_min": -width,
                            "n_max": width,
                        }
    return True


def configure_wam_terminations(config):
    """Select matching hard/soft failure rules for new physical WAM runs.

    Apply before serializing the training config and again before environment
    creation (also used by evaluation). PPO, flat WAM experiments, and old
    checkpoint configs without the alignment flag keep their existing rules.
    """
    target = str(OmegaConf.select(config, "trainer._target_", default=""))
    if target.rsplit(".", 1)[-1] != "WAMTrainer":
        return False
    if OmegaConf.select(config, "world_model.cfg.state_type", default="flat") != "physical":
        return False
    if not OmegaConf.select(config, "trainer.settings.align_real_termination", default=False):
        return False
    template = Path(__file__).resolve().parents[1] / "config/manager_env/terminations/tracking/wam.yaml"
    terminations = OmegaConf.load(template)
    terminations.wam_failure.params.height_threshold = float(OmegaConf.select(
        config, "trainer.settings.termination_height", default=0.25,
    ))
    terminations.wam_failure.params.tilt_threshold = float(OmegaConf.select(
        config, "trainer.settings.termination_tilt", default=0.8,
    ))
    with open_dict(config.manager_env):
        config.manager_env.terminations = terminations
    return True


def configure_wam_rewards(config):
    """Select one reward definition for physical WAM simulation and BPTT."""
    target = str(OmegaConf.select(config, "trainer._target_", default=""))
    if target.rsplit(".", 1)[-1] != "WAMTrainer":
        return False
    if OmegaConf.select(config, "world_model.cfg.state_type", default="flat") != "physical":
        return False
    if not OmegaConf.select(config, "trainer.settings.align_real_reward", default=False):
        return False
    config_dir = Path(__file__).resolve().parents[1] / "config"
    defaults = OmegaConf.load(config_dir / "world_model/wam_reward.yaml")
    overrides = OmegaConf.select(config, "trainer.reward_config", default={}) or {}
    reward_config = OmegaConf.merge(defaults, overrides)
    if reward_config.get("_target_") != defaults._target_:
        raise ValueError("Aligned WAM rewards require the shared WAMTrackingReward implementation")
    rewards = OmegaConf.load(config_dir / "manager_env/rewards/tracking/wam.yaml")
    parameters = OmegaConf.to_container(reward_config, resolve=True)
    parameters = {key: value for key, value in parameters.items() if not key.startswith("_")}
    terms = set(rewards) - {"_target_"}
    if set(parameters["weights"]) != terms:
        raise ValueError("WAM reward weights must match the eight physical tracking terms")
    for name in terms:
        rewards[name].weight = float(parameters["weights"][name])
        rewards[name].params.reward_config = parameters
    with open_dict(config.trainer):
        config.trainer.reward_config = reward_config
    with open_dict(config.manager_env):
        config.manager_env.rewards = rewards
    return True


def register_rl_resolvers():

    try:
        OmegaConf.register_new_resolver("eval", eval)
        OmegaConf.register_new_resolver("if", lambda pred, a, b: a if pred else b)
        OmegaConf.register_new_resolver("eq", lambda x, y: x.lower() == y.lower())
        OmegaConf.register_new_resolver("sqrt", lambda x: math.sqrt(float(x)))
        OmegaConf.register_new_resolver("sum", lambda x: sum(x))
        OmegaConf.register_new_resolver("ceil", lambda x: math.ceil(x))
        OmegaConf.register_new_resolver("int", lambda x: int(x))
        OmegaConf.register_new_resolver("len", lambda x: len(x))
        OmegaConf.register_new_resolver("sum_list", lambda lst: sum(lst))
    except Exception as e:
        logger.warning(f"Warning: Some resolvers already registered: {e}")
