"""验证 checkpoint 保存/加载"""
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import torch
from cleanmarl.algos import HAPPO
from cleanmarl.envs.cpdre_wrapper import make_cpdre_env

env_args = {
    "map_name": "direct_1c3u_A4_s42", "num_power": 3, "episode_len": 156,
    "weeks_per_year": 52, "off_peak_weeks": 26, "lead_time": 1, "seed": 42,
    "mechanism_mode": "none", "allocation_mode": "fair",
    "demand_mode": "deterministic", "price_mode": "seasonal",
    "demand_offpeak": 1.0, "demand_peak": 1.4, "demand_sigma": 0.0,
    "demand_ar1_rho": 0.60, "demand_corr": 0.0, "theta_offpeak": 1.15,
    "theta_peak": 0.80, "supply_sigma": 0.0, "price_offpeak": 1.0,
    "price_peak": 1.2, "price_sigma": 0.0, "price_feedback_kappa": 0.10,
    "price_feedback_psi": 0.70,
}
config = {
    "episode_len": 156, "rollout_length": 156, "num_sgd_iter": 1,
    "hidden_dim": 128, "device": "cuda" if torch.cuda.is_available() else "cpu",
    "log_dir": "./logs/ckpt_test", "experiment_name": "ckpt_test",
}

env = make_cpdre_env(env_args)
t = HAPPO(env, config)
t.current_step = 12345
t.current_epoch = 7

# 保存到不存在的目录（验证自动创建）
path = "./checkpoints/test_ckpt.pt"
t.save_checkpoint(path)
assert os.path.exists(path), "checkpoint文件未创建"

# 记录一个actor权重
w_before = t.actors[0].mean_head.weight.clone()

# 新建trainer并加载
t2 = HAPPO(env, config)
t2.load_checkpoint(path)
w_after = t2.actors[0].mean_head.weight

assert t2.current_step == 12345, f"step未恢复: {t2.current_step}"
assert t2.current_epoch == 7, f"epoch未恢复: {t2.current_epoch}"
assert torch.allclose(w_before, w_after), "actor权重未正确恢复"

print("✅ checkpoint 保存/加载验证通过")
print(f"   step={t2.current_step}, epoch={t2.current_epoch}, 权重一致={torch.allclose(w_before, w_after)}")

# 清理
os.remove(path)
env.close()
