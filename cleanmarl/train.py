"""
通用训练入口 - 从YAML配置文件读取所有参数

用法：
    python train.py --config configs/default.yaml
    python train.py --config configs/happo_cpdre.yaml
"""
import sys
import yaml
import argparse
from pathlib import Path
from typing import Dict, Any

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from cleanmarl.algos import HAPPO, MAPPO, IPPO
from cleanmarl.envs.cpdre_wrapper import make_cpdre_env


def load_config(path: str) -> Dict[str, Any]:
    """加载YAML配置文件"""
    with open(path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def merge_config(base: Dict, override: Dict) -> Dict:
    """递归合并配置（override覆盖base）"""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_config(result[key], value)
        else:
            result[key] = value
    return result


def build_policy(name: str, env, config: Dict):
    """根据策略名称创建训练器或规则策略。

    学习型: happo / mappo / ippo -> Trainer 子类。
    规则型: fixed / rule_a1 / rule_a2 / rule_b1 / rule_b3 / rule_b4 -> RulePolicy。
    """
    algos = {
        'happo': HAPPO,
        'mappo': MAPPO,
        'ippo': IPPO,
    }
    if name in algos:
        return algos[name](env, config)
    # Rule / fixed baselines (no gradient training).
    from cleanmarl.algos.rule_policy import RulePolicy
    rule_variants = {'fixed', 'rule_a1', 'rule_a2', 'rule_b1', 'rule_b3', 'rule_b4'}
    if name in rule_variants:
        return RulePolicy(env, config, variant=name)
    raise ValueError(f"未知策略: {name}。可选: {list(algos.keys()) + sorted(rule_variants)}")


def main():
    parser = argparse.ArgumentParser(description="CleanMARL 训练入口")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="YAML配置文件路径（相对于项目根目录）")
    parser.add_argument("--algo", type=str, default=None,
                        help="覆盖配置中的算法名 (happo/mappo/ippo)")
    parser.add_argument("--seed", type=int, default=None,
                        help="覆盖随机种子")
    parser.add_argument("--device", type=str, default=None,
                        help="覆盖设备 (cuda/cpu)")
    parser.add_argument("--timesteps", type=int, default=None,
                        help="覆盖总训练步数")
    args = parser.parse_args()

    # 加载配置
    config_path = project_root / "cleanmarl" / args.config
    cfg = load_config(config_path)

    # 构建训练配置（扁平化）
    train_config = {
        'episode_len': cfg['train']['episode_len'],
        'rollout_length': cfg['train']['rollout_length'],
        'num_sgd_iter': cfg['train']['num_sgd_iter'],
        'log_freq': cfg['train']['log_freq'],
        'gamma': cfg['ppo']['gamma'],
        'gae_lambda': cfg['ppo']['gae_lambda'],
        'clip_param': cfg['ppo']['clip_param'],
        'value_clip_param': cfg['ppo']['value_clip_param'],
        'entropy_coef': cfg['ppo']['entropy_coef'],
        'value_loss_coef': cfg['ppo']['value_loss_coef'],
        'max_grad_norm': cfg['ppo']['max_grad_norm'],
        'actor_lr': cfg['lr']['actor_lr'],
        'critic_lr': cfg['lr']['critic_lr'],
        'hidden_dim': cfg['model']['hidden_dim'],
        'rnn_layers': cfg['model'].get('rnn_layers', 1),
        'chunk_len': cfg['train'].get('chunk_len', 0),
        'device': cfg['system']['device'],
        'log_dir': cfg['system']['log_dir'],
        'experiment_name': cfg['experiment']['name'],
        'total_timesteps': cfg['train']['total_timesteps'],
        'log_step_details': cfg['train'].get('log_step_details', False),  # 新增
        'eval_episodes': cfg['train'].get('eval_episodes', 64),
        'eval_seed_offset': cfg['env'].get('eval_seed_offset', 9000),
    }

    # 命令行覆盖
    if args.algo:
        cfg['experiment']['algo'] = args.algo
    if args.seed is not None:
        cfg['env']['seed'] = args.seed
        train_config['experiment_name'] = f"{cfg['experiment']['algo']}_seed{args.seed}"
    if args.device:
        train_config['device'] = args.device
    if args.timesteps:
        train_config['total_timesteps'] = args.timesteps

    # 自动检测GPU
    if train_config['device'] == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA 不可用，切换到 CPU")
        train_config['device'] = 'cpu'

    # 打印配置
    print("=" * 80)
    print(f"CleanMARL - {cfg['experiment']['algo'].upper()} Training")
    print("=" * 80)
    print(f"\n📋 配置文件: {args.config}")
    print(f"   算法: {cfg['experiment']['algo']}")
    print(f"   环境: {cfg['env']['map_name']}")
    print(f"   总步数: {train_config['total_timesteps']}")
    print(f"   Rollout长度: {train_config['rollout_length']}")
    print(f"   Actor LR: {train_config['actor_lr']}")
    print(f"   Critic LR: {train_config['critic_lr']}")
    print(f"   设备: {train_config['device']}")
    print(f"   种子: {cfg['env']['seed']}")
    print()

    # 创建环境
    print("🌍 创建环境...")
    env = make_cpdre_env(cfg['env'])
    print(f"   Num agents: {env.num_agents}")
    print(f"   Obs dim: {env.observation_space.shape[0]}")
    print(f"   State dim: {env.state_space.shape[0]}")
    print(f"   Action dims: {env.act_dims} (max={env.max_act_dim}, continuous={env.continuous})")
    print()

    # 创建训练器/策略
    print(f"🤖 初始化 {cfg['experiment']['algo']} ...")
    trainer = build_policy(cfg['experiment']['algo'], env, train_config)
    print("   ✓ 已创建")
    print()

    # 开始训练
    try:
        trainer.train(
            total_timesteps=train_config['total_timesteps'],
            eval_freq=cfg['train']['eval_freq']
        )
    except KeyboardInterrupt:
        print("\n⚠️  训练被用户中断")
    finally:
        save_path = f"{cfg['system']['checkpoint_dir']}/{train_config['experiment_name']}_final.pt"
        trainer.save_checkpoint(save_path)
        env.close()

    print(f"\n✅ 完成。日志: {train_config['log_dir']}/{train_config['experiment_name']}/progress.csv")


if __name__ == "__main__":
    main()
