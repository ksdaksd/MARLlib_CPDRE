"""
PyTorch工具函数
"""
import torch
import numpy as np
import random


def set_seed(seed: int):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def explained_variance(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """
    计算explained variance

    EV = 1 - Var(y_true - y_pred) / Var(y_true)
    """
    var_y = torch.var(y_true)
    if var_y < 1e-8:
        return 0.0
    return 1 - torch.var(y_true - y_pred) / var_y


def huber_loss(x: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    """Huber loss"""
    return torch.where(
        torch.abs(x) < delta,
        0.5 * x.pow(2),
        delta * (torch.abs(x) - 0.5 * delta)
    )
