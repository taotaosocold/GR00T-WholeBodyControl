import math
from pathlib import Path

from loguru import logger
from omegaconf import OmegaConf, open_dict

from gear_sonic.utils.wam_noise import wam_observation_noise


def configure_wam_context(config):
    """Enable detailed simulator context only for the physical WAM branch."""
    target = str(OmegaConf.select(config, "trainer._target_", default=""))
    if target.rsplit(".", 1)[-1] != "WAMTrainer":
        return False
    if OmegaConf.select(config, "world_model.cfg.state_type", default="flat") != "physical":
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
    identical per-term corruption. This helper is inactive for PPO/flat WAM.
    """
    target = str(OmegaConf.select(config, "trainer._target_", default=""))
    if target.rsplit(".", 1)[-1] != "WAMTrainer":
        return False
    if OmegaConf.select(config, "world_model.cfg.state_type", default="flat") != "physical":
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
