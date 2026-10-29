import math
from pathlib import Path

from loguru import logger
from omegaconf import OmegaConf, open_dict


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
