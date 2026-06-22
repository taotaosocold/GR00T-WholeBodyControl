from omegaconf import DictConfig
from hydra.utils import instantiate

# 先用工厂函数来构建骨骼，在用工厂函数来初始化，可以通过motion_rep来获得运动序列的各个值
def load_motion_rep(conf: DictConfig):
    skeleton = instantiate(conf.skeleton)
    motion_rep = instantiate(conf.motion_rep, fps=conf.fps, skeleton=skeleton)
    return motion_rep
