"""VQVAE training script using synthetic data.

Demonstrates how the VQVAE training pipeline works without requiring
the actual motion dataset. Loads the saved model config from the
checkpoint directory and trains on randomly generated motion tensors.

Usage:
    python scripts/train_vqvae.py --max_steps 100
"""

import argparse
import copy
import os
# pytorch_lightning是一个现成的工具,无需自己实现训练循环、分布式训练、日志记录等底层细节
import pytorch_lightning as pl
import torch
from functools import partial
# instantiate是工厂函数,根据配置文件的_target_来知道是实例化的类在哪里,其接受的参数是"一个配置文件"+"多个去覆盖配置文件的参数"
from hydra.utils import instantiate
from omegaconf import OmegaConf, open_dict
from torch.utils.data import DataLoader

from motionbricks.data.synthetic_dataset import SyntheticMotionDataset, collate_batch
from motionbricks.helper.pl_util import load_motion_rep

# 指定的配置文件就是hparams.yaml
def load_config(result_dir: str, max_steps: int):
    """Load and patch hparams.yaml for single-GPU training."""
    version_dir = os.path.join(result_dir, "motionbricks_vqvae", "version_1")
    hparams_path = os.path.join(version_dir, "hparams.yaml")
    # 根据对应yaml文件来配置配置文件config
    conf = OmegaConf.load(hparams_path)

    with open_dict(conf):
        # resolve data paths to the version directory (where skeleton/stats live)
        conf.data = {"folder": version_dir}
        conf.skeleton.folder = os.path.join(version_dir, "skeleton")
        conf.motion_rep.stats.folder = os.path.join(version_dir, "stats", "motion")

        # single-GPU training overrides
        conf.trainer.devices = 1
        conf.trainer.num_nodes = 1
        conf.trainer.max_steps = max_steps
        conf.trainer.accelerator = "auto"
        conf.trainer.strategy = "auto"
        conf.trainer.enable_progress_bar = True
        conf.trainer.log_every_n_steps = 10
        conf.trainer.val_check_interval = max_steps  # no validation
        conf.trainer.num_sanity_val_steps = 0

        # resolve ${trainer.max_steps} in scheduler
        conf.model.scheduler.num_training_steps = max_steps

    return conf, version_dir


def main():
    parser = argparse.ArgumentParser(description="VQVAE training")
    # 存放预训练检查点（及配置）的根目录
    parser.add_argument("--result_dir", type=str, default="./out",
                        help="Directory containing pretrained checkpoints")
    # 训练步数，默认 200
    parser.add_argument("--max_steps", type=int, default=200,
                        help="Number of training steps")
    # 批大小，默认 8
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size")
    # 合成数据集的样本数量，默认 500
    parser.add_argument("--num_samples", type=int, default=500,
                        help="Number of synthetic samples in dataset")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pl.seed_everything(args.seed)
    # 调用 load_config 得到配置和配置文件所在目录的路径
    conf, version_dir = load_config(args.result_dir, args.max_steps)

    # instantiate skeleton and motion representation
    # 根据配置文件构建运动表示对象，这个load_motion_rep会通过初始化骨胳以及前向运动学来计算运动序列的各个全局或局部值（默认是全局值），就可以直接通过motion_rep获得了
    motion_rep = load_motion_rep(conf)
    # 返回所有需要计算的特征索引列表，其长度就是每条运动帧的特征维度
    feat_dim = len(motion_rep.indices['all'])

    # create synthetic dataset
    # min_frames must exceed max possible num_frames + 1 used in training_step
    # max_tokens=16, down_t=2 => max frames = 16 * 4 = 64, +1 for global->local = 65
    # 实例化合成数据集。注释解释了 min_frames 设为 80 的原因：VQVAE 的某些操作（如全局到局部变换）需要至少 65 帧，这里设置更大的 80 以确保安全。
    dataset = SyntheticMotionDataset(
        feat_dim=feat_dim,
        num_samples=args.num_samples,
        min_frames=80,
        max_frames=200,
    )
    # 数据集会为每个样本生成一个随机的运动序列，形状为 (L, feat_dim)，L 在 80~200 之间随机。
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_batch,   # collate_batch 是自定义函数，用于将不同长度的序列填充成相同长度，并生成 mask 等
        persistent_workers=True,    # 保持 worker 进程存活，避免每个 epoch 重复启动开销
    )

    # instantiate the VQVAE network and model
    # 拷贝配置文件中的模型配置部分
    model_conf = copy.deepcopy(conf.model)
    with open_dict(model_conf):
        # inject the motion_rep into sub-configs that use ???
        # 在 model_conf 中，pose_vqvae_network 子配置可能有需要 motion_rep 参数的占位符（???），这里实例化时传入局部运动表示 motion_rep.dual_rep.local_motion_rep。得到姿势 VQ-VAE 的网络结构
        # 在这里pose_net是单纯的网络,其继承的是nn.Module其是vqvae,内含编码,解码,码本等
        pose_net = instantiate(
            model_conf.pose_vqvae_network,
            motion_rep=motion_rep.dual_rep.local_motion_rep,
        )
        # build optimizer and scheduler as partials
        # 用工厂函数实例化优化器配置和调度器配置
        optimizer_fn = instantiate(model_conf.optimizer)
        scheduler_fn = instantiate(model_conf.scheduler) if model_conf.scheduler else None
        # 用 instantiate 创建整个 VQVAE 模型
        # 这个是完整的vqvae模型,其和pose_net的区别相当于其继承pose_net但是又多了很多新的函数,包括反向传播之类的函数.
        model = instantiate(
            model_conf,
            pose_vqvae_network=pose_net,
            root_vqvae_network=None,
            motion_rep=motion_rep,
            optimizer=optimizer_fn,
            scheduler=scheduler_fn,
            _recursive_=False,
        )

    # create trainer (no callbacks needed)
    # 直接使用第三方工具来创建训练器
    trainer = pl.Trainer(
        max_steps=conf.trainer.max_steps,
        devices=conf.trainer.devices,       # 使用的 GPU 数量，这里强制为 1
        num_nodes=conf.trainer.num_nodes,   # 节点数（分布式训练用），这里也是 1
        accelerator=conf.trainer.accelerator,   # 硬件加速器类型（如 "gpu", "cpu"），设为 "auto" 让 Lightning 自动检测
        strategy=conf.trainer.strategy, # 分布式策略（如 "ddp"），设为 "auto" 自动选择
        precision=conf.trainer.precision,   # 混合精度训练（如 "16-mixed" 或 32），从配置读取
        gradient_clip_val=conf.trainer.gradient_clip_val,   # 梯度裁剪阈值，从配置读取
        enable_progress_bar=conf.trainer.enable_progress_bar,   # 是否显示进度条，来自配置（True）
        log_every_n_steps=conf.trainer.log_every_n_steps,   # 每隔多少步记录一次训练指标（如 loss），来自配置（10）
        num_sanity_val_steps=0,     # 不做初始验证
        enable_checkpointing=False, # 不保存模型检查点（因为只是演示）
        logger=False,           # 不使用任何日志记录器（如 TensorBoard），只依赖进度条打印
    )

    print(f"Starting VQVAE training for {args.max_steps} steps...")
    print(f"  Feature dim: {feat_dim}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Dataset size: {args.num_samples}")
    trainer.fit(model, train_dataloaders=dataloader)
    print("Training complete.")


if __name__ == "__main__":
    main()
