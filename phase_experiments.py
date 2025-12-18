# project_main.py
# ============================================================
# Colored MNIST Diffusion Project (Phase 1–4) — single-file, автономно
#
# Что умеет:
#  - Train DDPM (ε-pred) на Colored MNIST
#  - Phase 1: DEIS-like sampler (logSNR grid + exp-integrator idea + poly extrapolation)
#  - Phase 2: ES-DDPM (early-stop reverse) + "forward early-stop" reconstruction benchmark
#  - Phase 3: DAED-like analysis (SNR/logSNR + noise/x0 error vs t, two-phase style)
#  - Phase 4: HSIVI-like sampler (hierarchical semi-implicit mixture over latent perturbations)
#  - Speed benchmarks + speedups + графики
#  - Quality proxies (MSE/PSNR по t, reconstruction PSNR, распределения ошибок)
#
# Примеры:
#   1) Обучение:
#      python3 project_main.py --mode train --outdir results_train --schedule cosine --epochs 5 --use_snr_weighted
#
#   2) Полный прогон (анализ+скорость+метрики) на готовом чекпойнте:
#      python3 project_main.py --mode eval --ckpt results/model_epoch5.pth --outdir exp_run --schedule cosine
#
#   3) Только бенчмарк скорости:
#      python3 project_main.py --mode bench --ckpt results/model_epoch5.pth --outdir exp_run --schedule cosine
#
#   4) Только анализ (SNR/ошибки/forward-early-stop):
#      python3 project_main.py --mode analyze --ckpt results/model_epoch5.pth --outdir exp_run --schedule cosine
#
# ============================================================

import os
import time
import math
import json
import csv
import argparse
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torchvision import datasets, transforms
from torchvision.utils import save_image

import matplotlib.pyplot as plt


# =========================
#  Config
# =========================

@dataclass
class DiffusionConfig:
    image_size: int = 28
    channels: int = 3
    T: int = 1000

    beta_start: float = 1e-4
    beta_end: float = 0.02
    schedule: str = "cosine"  # "cosine" | "linear"

    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
#  Utils
# =========================

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    import random
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def denorm(x: torch.Tensor) -> torch.Tensor:
    # [-1, 1] -> [0, 1]
    return (x + 1.0) * 0.5


def now_run_id() -> str:
    # локально, без таймзон-сложностей
    return time.strftime("%Y%m%d_%H%M%S")


def cuda_sync_if_needed(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def psnr_from_mse(mse: torch.Tensor, data_range: float = 2.0, eps: float = 1e-12) -> torch.Tensor:
    # если значения в [-1,1], то data_range=2.0
    # PSNR = 10 log10( data_range^2 / mse )
    return 10.0 * torch.log10((data_range * data_range) / (mse + eps))


# =========================
#  Dataset: Colored MNIST
# =========================

class ColoredMNIST(Dataset):
    """
    Colored MNIST:
      - MNIST (1x28x28) -> повторяем до 3 каналов
      - домножаем цифру на случайный цвет (фон остается темнее)
      - приводим в [-1, 1]
    """
    def __init__(self, root: str, train: bool):
        self.base = datasets.MNIST(
            root=root,
            train=train,
            download=True,
            transform=transforms.ToTensor(),
        )

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        x, y = self.base[idx]        # [1,28,28] in [0,1]
        x = x.repeat(3, 1, 1)        # [3,28,28]
        color = torch.rand(3, 1, 1) * 0.8 + 0.2
        x = x * color
        x = x * 2.0 - 1.0            # [-1,1]
        return x, y


# =========================
#  Schedules + diffusion params
# =========================

def cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    steps = T + 1
    x = torch.linspace(0, T, steps)
    alphas_cumprod = torch.cos(((x / T) + s) / (1 + s) * math.pi / 2.0) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 1e-8, 0.999)


def linear_beta_schedule(T: int, beta_start: float, beta_end: float) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, T)


def make_diffusion_params(cfg: DiffusionConfig) -> Dict[str, torch.Tensor]:
    if cfg.schedule == "cosine":
        betas = cosine_beta_schedule(cfg.T)
    else:
        betas = linear_beta_schedule(cfg.T, cfg.beta_start, cfg.beta_end)

    betas = betas.to(cfg.device)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)

    alpha_bar_prev = torch.cat(
        [torch.tensor([1.0], device=cfg.device), alpha_bar[:-1]], dim=0
    )

    sqrt_alpha_bar = torch.sqrt(alpha_bar)
    sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar)

    posterior_variance = betas * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)

    return {
        "betas": betas,
        "alphas": alphas,
        "alpha_bar": alpha_bar,
        "alpha_bar_prev": alpha_bar_prev,
        "sqrt_alpha_bar": sqrt_alpha_bar,
        "sqrt_one_minus_alpha_bar": sqrt_one_minus_alpha_bar,
        "posterior_variance": posterior_variance,
    }


def snr_from_alpha_bar(alpha_bar: torch.Tensor) -> torch.Tensor:
    return alpha_bar / (1.0 - alpha_bar)


def log_snr_from_alpha_bar(alpha_bar: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    snr = snr_from_alpha_bar(alpha_bar)
    return torch.log(snr + eps)


# =========================
#  Time embedding
# =========================

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: [B]
        half = self.dim // 2
        # частоты
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(0, half, device=t.device) / max(half - 1, 1)
        )
        args = t.float()[:, None] * freqs[None]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        return emb


# =========================
#  U-Net (compact)
# =========================

class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Linear(time_dim, out_ch)
        self.act = nn.SiLU()
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        t = self.time_mlp(t_emb)[:, :, None, None]
        h = h + t
        h = self.act(h)
        h = self.conv2(h)
        return self.act(h + self.skip(x))


class SimpleUNet(nn.Module):
    """
    28x28: 28 -> 14 -> 7 -> 14 -> 28
    """
    def __init__(self, cfg: DiffusionConfig, time_dim: int = 128, base_ch: int = 64 ): # было 64
        super().__init__()
        self.cfg = cfg

        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )

        ch = base_ch
        self.inc = ResidualBlock(cfg.channels, ch, time_dim)       # 28
        self.down1 = ResidualBlock(ch, ch * 2, time_dim)          # 14
        self.down2 = ResidualBlock(ch * 2, ch * 4, time_dim)      # 7
        self.mid = ResidualBlock(ch * 4, ch * 4, time_dim)        # 7

        self.up1 = ResidualBlock(ch * 4 + ch * 2, ch * 2, time_dim)  # 14
        self.up2 = ResidualBlock(ch * 2 + ch, ch, time_dim)          # 28
        self.outc = nn.Conv2d(ch, cfg.channels, 1)

        self.pool = nn.AvgPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(t)
        x1 = self.inc(x, t_emb)                      # [B,ch,28,28]
        x2 = self.down1(self.pool(x1), t_emb)        # [B,2ch,14,14]
        x3 = self.down2(self.pool(x2), t_emb)        # [B,4ch,7,7]
        m = self.mid(x3, t_emb)                      # [B,4ch,7,7]

        u1 = self.up1(torch.cat([self.upsample(m), x2], dim=1), t_emb)   # [B,2ch,14,14]
        u2 = self.up2(torch.cat([self.upsample(u1), x1], dim=1), t_emb)  # [B,ch,28,28]
        return self.outc(u2)                                         # ε_pred


# =========================
#  DDPM Core
# =========================

class DDPM:
    def __init__(self, cfg: DiffusionConfig, params: Dict[str, torch.Tensor]):
        self.cfg = cfg
        self.params = params

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab = self.params["sqrt_alpha_bar"][t].view(-1, 1, 1, 1)
        sqrt_om = self.params["sqrt_one_minus_alpha_bar"][t].view(-1, 1, 1, 1)
        return sqrt_ab * x0 + sqrt_om * noise

    def predict_x0_from_eps(self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        sqrt_ab = self.params["sqrt_alpha_bar"][t].view(-1, 1, 1, 1)
        sqrt_om = self.params["sqrt_one_minus_alpha_bar"][t].view(-1, 1, 1, 1)
        x0 = (x_t - sqrt_om * eps) / (sqrt_ab + 1e-12)
        
        return x0.clamp(-1.0, 1.0)

    def p_mean_variance(self, model: nn.Module, x_t: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        betas = self.params["betas"]
        alpha_bar = self.params["alpha_bar"]
        alpha_bar_prev = self.params["alpha_bar_prev"]
        posterior_variance = self.params["posterior_variance"]
    
        beta_t = betas[t].view(-1, 1, 1, 1)
        alpha_t = (1.0 - beta_t)                         # [B,1,1,1]
        ab_t = alpha_bar[t].view(-1, 1, 1, 1)
        ab_prev = alpha_bar_prev[t].view(-1, 1, 1, 1)
        var = posterior_variance[t].view(-1, 1, 1, 1)
    
        eps_theta = model(x_t, t)
    
        # x0_pred from eps
        x0_pred = (x_t - torch.sqrt(1.0 - ab_t) * eps_theta) / (torch.sqrt(ab_t) + 1e-12)
        # (опционально, но очень помогает стабилизации и метрикам)
        x0_pred = x0_pred.clamp(-1.0, 1.0)
    
        coef_x0 = torch.sqrt(ab_prev) * beta_t / (1.0 - ab_t + 1e-12)
        coef_xt = torch.sqrt(alpha_t) * (1.0 - ab_prev) / (1.0 - ab_t + 1e-12)
    
        mean = coef_x0 * x0_pred + coef_xt * x_t
        return mean, var, eps_theta


# =========================
#  Losses
# =========================

def diffusion_loss_mse(model: nn.Module, ddpm: DDPM, x0: torch.Tensor) -> torch.Tensor:
    B = x0.size(0)
    device = x0.device
    T = ddpm.cfg.T
    t = torch.randint(0, T, (B,), device=device).long()
    noise = torch.randn_like(x0)
    x_t = ddpm.q_sample(x0, t, noise)
    eps_pred = model(x_t, t)
    return F.mse_loss(eps_pred, noise)


def diffusion_loss_snr_weighted(model: nn.Module, ddpm: DDPM, x0: torch.Tensor) -> torch.Tensor:
    B = x0.size(0)
    device = x0.device
    T = ddpm.cfg.T
    t = torch.randint(0, T, (B,), device=device).long()
    noise = torch.randn_like(x0)
    x_t = ddpm.q_sample(x0, t, noise)
    eps_pred = model(x_t, t)

    alpha_bar_t = ddpm.params["alpha_bar"][t]   # [B]
    snr_t = alpha_bar_t / (1.0 - alpha_bar_t + 1e-12)
    w_t = 1.0 / (1.0 + snr_t)

    per_pixel = (eps_pred - noise) ** 2
    per_sample = per_pixel.view(B, -1).mean(dim=1)
    return (w_t * per_sample).mean()


# =========================
#  Samplers (baseline)
# =========================

@torch.no_grad()
def sample_ddpm_full(model: nn.Module, ddpm: DDPM, batch_size: int) -> torch.Tensor:
    T = ddpm.cfg.T
    device = ddpm.cfg.device
    x_t = torch.randn(batch_size, ddpm.cfg.channels, ddpm.cfg.image_size, ddpm.cfg.image_size, device=device)

    model.eval()
    for t in range(T - 1, -1, -1):
        t_b = torch.full((batch_size,), t, device=device, dtype=torch.long)
        mean, var, _ = ddpm.p_mean_variance(model, x_t, t_b)
        if t > 0:
            x_t = mean + torch.sqrt(var) * torch.randn_like(x_t)
        else:
            x_t = mean
    model.train()
    return x_t


@torch.no_grad()
def sample_ddpm_coarse(model: nn.Module, ddpm: DDPM, batch_size: int, num_steps: int) -> torch.Tensor:
    T = ddpm.cfg.T
    device = ddpm.cfg.device
    chosen = torch.linspace(0, T - 1, num_steps, dtype=torch.long, device=device)
    chosen = torch.unique(chosen, sorted=True)
    timesteps = list(reversed(chosen.tolist()))

    x_t = torch.randn(batch_size, ddpm.cfg.channels, ddpm.cfg.image_size, ddpm.cfg.image_size, device=device)

    model.eval()
    for t_int in timesteps:
        t_b = torch.full((batch_size,), t_int, device=device, dtype=torch.long)
        mean, var, _ = ddpm.p_mean_variance(model, x_t, t_b)
        if t_int > 0:
            x_t = mean + torch.sqrt(var) * torch.randn_like(x_t)
        else:
            x_t = mean
    model.train()
    return x_t


@torch.no_grad()
def sample_ddim(model: nn.Module, ddpm: DDPM, batch_size: int, num_steps: int, eta: float = 0.0) -> torch.Tensor:
    device = ddpm.cfg.device
    T = ddpm.cfg.T
    alpha_bar = ddpm.params["alpha_bar"].to(device)

    chosen = torch.linspace(0, T - 1, num_steps, dtype=torch.long, device=device)
    chosen = torch.unique(chosen, sorted=True)
    timesteps = list(reversed(chosen.tolist()))

    x_t = torch.randn(batch_size, ddpm.cfg.channels, ddpm.cfg.image_size, ddpm.cfg.image_size, device=device)

    model.eval()
    for i, t_int in enumerate(timesteps):
        t_b = torch.full((batch_size,), t_int, device=device, dtype=torch.long)
        eps = model(x_t, t_b)

        ab_t = alpha_bar[t_int]
        sqrt_ab_t = torch.sqrt(ab_t)
        sqrt_om_t = torch.sqrt(1.0 - ab_t)

        x0 = (x_t - sqrt_om_t * eps) / (sqrt_ab_t + 1e-12)

        if i == len(timesteps) - 1:
            x_t = x0
            break

        t_next = timesteps[i + 1]
        ab_next = alpha_bar[t_next]
        sqrt_ab_next = torch.sqrt(ab_next)
        sqrt_om_next = torch.sqrt(1.0 - ab_next)

        if eta == 0.0:
            x_t = sqrt_ab_next * x0 + sqrt_om_next * eps
        else:
            sigma = eta * torch.sqrt((1 - ab_next) / (1 - ab_t + 1e-12) * (1 - ab_t / (ab_next + 1e-12)))
            x_t = (
                sqrt_ab_next * x0
                + torch.sqrt(torch.clamp(1 - ab_next - sigma * sigma, min=0.0)) * eps
                + sigma * torch.randn_like(x_t)
            )
    model.train()
    return x_t


# =========================
#  Phase 1: DEIS-like (exp-integrator idea + poly extrap.)
# =========================

@torch.no_grad()
def sample_deis_like(model: nn.Module, ddpm: DDPM, batch_size: int, num_steps: int) -> torch.Tensor:
    """
    DEIS-like с 2nd order polynomial extrapolation:
      - равномерная сетка по lambda = logSNR
      - DDIM-ODE шаг (eta=0)
      - 2nd order polynomial extrapolation по x0:
            если history пуст: x0_deis = x0
            если 1 элемент: x0_deis = 2*x0 - x0_history[-1]
            если 2+ элемента: x0_deis = 3*x0 - 3*x0_history[-1] + x0_history[-2]
    """
    device = ddpm.cfg.device
    alpha_bar = ddpm.params["alpha_bar"].to(device)
    log_snr = log_snr_from_alpha_bar(alpha_bar)

    lam_T = log_snr[-1]
    lam_0 = log_snr[0]
    lam_grid = torch.linspace(lam_T, lam_0, num_steps, device=device)

    # индексы t, ближайшие к lambda-grid
    t_indices: List[int] = []
    for lam in lam_grid:
        idx = torch.argmin(torch.abs(log_snr - lam)).item()
        t_indices.append(idx)
    t_indices = sorted(set(t_indices), reverse=True)

    x_t = torch.randn(batch_size, ddpm.cfg.channels, ddpm.cfg.image_size, ddpm.cfg.image_size, device=device)

    model.eval()
    x0_history: List[torch.Tensor] = []  # храним историю x0 для экстраполяции

    for i, t_int in enumerate(t_indices):
        t_b = torch.full((batch_size,), t_int, device=device, dtype=torch.long)
        eps = model(x_t, t_b)

        ab_t = alpha_bar[t_int]
        sqrt_ab_t = torch.sqrt(ab_t)
        sqrt_om_t = torch.sqrt(1.0 - ab_t)

        x0 = (x_t - sqrt_om_t * eps) / (sqrt_ab_t + 1e-12)

        # 2nd order polynomial extrapolation
        if len(x0_history) == 0:
            x0_deis = x0
        elif len(x0_history) == 1:
            # 1st order: линейная экстраполяция
            x0_deis = 2.0 * x0 - x0_history[-1]
        else:
            # 2nd order: квадратичная экстраполяция
            x0_deis = 3.0 * x0 - 3.0 * x0_history[-1] + x0_history[-2]

        # обновляем историю (храним только 2 последних)
        x0_history.append(x0.detach())
        if len(x0_history) > 2:
            x0_history.pop(0)

        if i == len(t_indices) - 1:
            x_t = x0_deis
            break

        t_next = t_indices[i + 1]
        ab_next = alpha_bar[t_next]
        sqrt_ab_next = torch.sqrt(ab_next)
        sqrt_om_next = torch.sqrt(1.0 - ab_next)

        x_t = sqrt_ab_next * x0_deis + sqrt_om_next * eps

    model.train()
    return x_t


# =========================
#  "ExplInt" (logSNR grid, but without extrapolation)
# =========================

@torch.no_grad()
def sample_explint(model: nn.Module, ddpm: DDPM, batch_size: int, num_steps: int) -> torch.Tensor:
    device = ddpm.cfg.device
    alpha_bar = ddpm.params["alpha_bar"].to(device)
    log_snr = log_snr_from_alpha_bar(alpha_bar)

    lam_T = log_snr[-1]
    lam_0 = log_snr[0]
    lam_grid = torch.linspace(lam_T, lam_0, num_steps, device=device)

    t_indices: List[int] = []
    for lam in lam_grid:
        idx = torch.argmin(torch.abs(log_snr - lam)).item()
        t_indices.append(idx)
    t_indices = sorted(set(t_indices), reverse=True)

    x_t = torch.randn(batch_size, ddpm.cfg.channels, ddpm.cfg.image_size, ddpm.cfg.image_size, device=device)

    model.eval()
    for i, t_int in enumerate(t_indices):
        t_b = torch.full((batch_size,), t_int, device=device, dtype=torch.long)
        eps = model(x_t, t_b)

        ab_t = alpha_bar[t_int]
        sqrt_ab_t = torch.sqrt(ab_t)
        sqrt_om_t = torch.sqrt(1.0 - ab_t)
        x0 = (x_t - sqrt_om_t * eps) / (sqrt_ab_t + 1e-12)

        if i == len(t_indices) - 1:
            x_t = x0
            break

        t_next = t_indices[i + 1]
        ab_next = alpha_bar[t_next]
        sqrt_ab_next = torch.sqrt(ab_next)
        sqrt_om_next = torch.sqrt(1.0 - ab_next)
        x_t = sqrt_ab_next * x0 + sqrt_om_next * eps

    model.train()
    return x_t


# =========================
#  Phase 2: ES-DDPM (early-stop reverse)
# =========================

@torch.no_grad()
def sample_es_ddpm(model: nn.Module, ddpm: DDPM, batch_size: int, t_stop: int, num_steps: int) -> torch.Tensor:
    """
    ES-DDPM (правильная версия):
      - разреженная сетка t_stop -> 0 (num_steps)
      - стохастический reverse (DDPM-style noise)
    """
    device = ddpm.cfg.device
    T = ddpm.cfg.T
    t_stop = int(min(max(t_stop, 1), T - 1))
    
    # Сетка шагов от t_stop до 0 (разреженная)
    chosen = torch.linspace(t_stop, 0, num_steps, dtype=torch.long, device=device)
    chosen = torch.unique(chosen, sorted=True)
    timesteps = list(reversed(chosen.tolist()))
    
    x_t = torch.randn(batch_size, ddpm.cfg.channels, ddpm.cfg.image_size, ddpm.cfg.image_size, device=device)
    
    model.eval()
    for i, t_int in enumerate(timesteps):
        t_b = torch.full((batch_size,), t_int, device=device, dtype=torch.long)
        mean, var, _ = ddpm.p_mean_variance(model, x_t, t_b)
        
        if i < len(timesteps) - 1:  # не последний шаг
            x_t = mean + torch.sqrt(var) * torch.randn_like(x_t)
        else:
            x_t = mean
        
        x_t = x_t.clamp(-5.0, 5.0)
    
    model.train()
    return x_t



# =========================
#  Phase 4: HSIVI-like sampler (hierarchical semi-implicit mixture)
# =========================

@torch.no_grad()
def sample_hsivi(model: nn.Module, ddpm: DDPM, batch_size: int, num_steps: int, K: int = 4, perturb_scale: float = 0.15) -> torch.Tensor:
    """
    HSIVI-like:
      - на каждом шаге t делаем K "латентных" возмущений z_k (semi-implicit mixture)
      - усредняем x0_k и eps_k
      - делаем DDIM-ODE шаг по среднему (eta=0)
    """
    device = ddpm.cfg.device
    T = ddpm.cfg.T
    alpha_bar = ddpm.params["alpha_bar"].to(device)

    chosen = torch.linspace(0, T - 1, num_steps, dtype=torch.long, device=device)
    chosen = torch.unique(chosen, sorted=True)
    timesteps = list(reversed(chosen.tolist()))

    x_t = torch.randn(batch_size, ddpm.cfg.channels, ddpm.cfg.image_size, ddpm.cfg.image_size, device=device)

    model.eval()
    for i, t_int in enumerate(timesteps):
        t_int = int(t_int)
        t_b = torch.full((batch_size,), t_int, device=device, dtype=torch.long)

        ab_t = alpha_bar[t_int]
        sqrt_ab_t = torch.sqrt(ab_t)
        sqrt_om_t = torch.sqrt(1.0 - ab_t)

        log_weights = []
        x0_list = []
        eps_list = []

        for _ in range(int(K)):
            z = perturb_scale * torch.randn_like(x_t)
            x_tk = x_t + z
            eps_k = model(x_tk, t_b)
            x0_k = (x_tk - sqrt_om_t * eps_k) / (sqrt_ab_t + 1e-12)
            
            # log p(x_t | x0_k) как вес (Gaussian likelihood)
            var_perturbation = perturb_scale ** 2
            log_p = -0.5 * torch.sum((x_t - x_tk)**2 / var_perturbation, dim=[1,2,3])
            log_weights.append(log_p)
            x0_list.append(x0_k)
            eps_list.append(eps_k)

        # weighted average
        log_w = torch.stack(log_weights, dim=0)  # [K, B]
        w = F.softmax(log_w, dim=0)              # [K, B]
        
        x0_mix = sum(w[k, :, None, None, None] * x0_list[k] for k in range(K))
        eps_mix = sum(w[k, :, None, None, None] * eps_list[k] for k in range(K))

        if i == len(timesteps) - 1:
            x_t = x0_mix
            break

        t_next = int(timesteps[i + 1])
        ab_next = alpha_bar[t_next]
        sqrt_ab_next = torch.sqrt(ab_next)
        sqrt_om_next = torch.sqrt(1.0 - ab_next)

        x_t = sqrt_ab_next * x0_mix + sqrt_om_next * eps_mix

    model.train()
    return x_t


# =========================
#  Training
# =========================

def train(cfg: DiffusionConfig,
          outdir: str,
          epochs: int,
          batch_size: int,
          lr: float,
          use_snr_weighted: bool,
          num_workers: int = 0) -> str:
    device = cfg.device
    set_seed(cfg.seed)

    run_dir = ensure_dir(os.path.join(outdir, f"train_{now_run_id()}"))
    ckpt_dir = ensure_dir(os.path.join(run_dir, "checkpoints"))
    sample_dir = ensure_dir(os.path.join(run_dir, "samples"))
    plot_dir = ensure_dir(os.path.join(run_dir, "plots"))

    ds = ColoredMNIST(root=os.path.join(run_dir, "data"), train=True)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)

    params = make_diffusion_params(cfg)
    ddpm = DDPM(cfg, params)

    model = SimpleUNet(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    # SNR plots (schedule)
    alpha_bar_cpu = params["alpha_bar"].detach().cpu()
    snr = snr_from_alpha_bar(alpha_bar_cpu).numpy()
    log_snr = log_snr_from_alpha_bar(alpha_bar_cpu).numpy()

    plt.figure()
    plt.plot(snr)
    plt.xlabel("t")
    plt.ylabel("SNR(t)")
    plt.title("SNR schedule")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "snr_schedule.png"))
    plt.close()

    plt.figure()
    plt.plot(log_snr)
    plt.xlabel("t")
    plt.ylabel("log SNR(t)")
    plt.title("log SNR schedule")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "log_snr_schedule.png"))
    plt.close()

    loss_hist: List[float] = []
    global_step = 0

    for ep in range(epochs):
        model.train()
        for x, _ in dl:
            x = x.to(device)
            if use_snr_weighted:
                loss = diffusion_loss_snr_weighted(model, ddpm, x)
            else:
                loss = diffusion_loss_mse(model, ddpm, x)

            opt.zero_grad()
            loss.backward()
            opt.step()

            loss_hist.append(float(loss.item()))
            global_step += 1

            if global_step % 100 == 0:
                print(f"epoch {ep+1}/{epochs} step {global_step} loss={loss.item():.5f}")

        # samples after epoch (small)
        model.eval()
        with torch.no_grad():
            s_ddim = sample_ddim(model, ddpm, batch_size=16, num_steps=50, eta=0.0)
            s_deis = sample_deis_like(model, ddpm, batch_size=16, num_steps=50)
            s_es = sample_es_ddpm(model, ddpm, batch_size=16, t_stop=400, num_steps=50)
            s_hs = sample_hsivi(model, ddpm, batch_size=16, num_steps=50, K=4)

        save_image(denorm(s_ddim), os.path.join(sample_dir, f"ddim50_epoch{ep+1}.png"), nrow=4)
        save_image(denorm(s_deis), os.path.join(sample_dir, f"deis50_epoch{ep+1}.png"), nrow=4)
        save_image(denorm(s_es), os.path.join(sample_dir, f"es_t400_50_epoch{ep+1}.png"), nrow=4)
        save_image(denorm(s_hs), os.path.join(sample_dir, f"hsivi50_k4_epoch{ep+1}.png"), nrow=4)

        ckpt_path = os.path.join(ckpt_dir, f"model_epoch{ep+1}.pth")
        torch.save(model.state_dict(), ckpt_path)
        print("saved:", ckpt_path)

    # loss plot
    plt.figure()
    plt.plot(loss_hist)
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title("training loss")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "training_loss.png"))
    plt.close()

    # summary
    with open(os.path.join(run_dir, "train_config.json"), "w") as f:
        json.dump({
            "cfg": cfg.__dict__,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "use_snr_weighted": bool(use_snr_weighted),
        }, f, indent=2)

    print("train done:", run_dir)
    return run_dir


# =========================
#  Load model
# =========================

def load_model(cfg: DiffusionConfig, ckpt_path: str) -> Tuple[DDPM, nn.Module]:
    params = make_diffusion_params(cfg)
    ddpm = DDPM(cfg, params)
    model = SimpleUNet(cfg).to(cfg.device)
    state = torch.load(ckpt_path, map_location=cfg.device)
    model.load_state_dict(state)
    model.eval()
    return ddpm, model


# =========================
#  Analysis: SNR / DAED-like metrics
# =========================

@torch.no_grad()
def analyze_snr(ddpm: DDPM, outdir: str) -> None:
    outdir = ensure_dir(outdir)
    alpha_bar = ddpm.params["alpha_bar"].detach().cpu()
    snr = snr_from_alpha_bar(alpha_bar).numpy()
    log_snr = log_snr_from_alpha_bar(alpha_bar).numpy()

    plt.figure()
    plt.plot(snr)
    plt.xlabel("t")
    plt.ylabel("SNR(t)")
    plt.title("SNR schedule")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "snr_schedule.png"))
    plt.close()

    plt.figure()
    plt.plot(log_snr)
    plt.xlabel("t")
    plt.ylabel("log SNR(t)")
    plt.title("log SNR schedule")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "log_snr_schedule.png"))
    plt.close()

    with open(os.path.join(outdir, "snr_values.json"), "w") as f:
        json.dump({"snr": snr.tolist(), "log_snr": log_snr.tolist()}, f, indent=2)


@torch.no_grad()
def eval_error_vs_t(cfg: DiffusionConfig,
                    ddpm: DDPM,
                    model: nn.Module,
                    outdir: str,
                    split: str,
                    num_t_points: int,
                    max_batches: int,
                    batch_size: int,
                    num_workers: int = 0) -> Dict[str, List[float]]:
    """
    DAED-like:
      - MSE(ε_pred, ε_true) vs t
      - MSE(x0_pred, x0_true) vs t
      - PSNR(x0_pred, x0_true) vs t
      - Weighted variants with w(t)=1/(1+SNR)
    """
    outdir = ensure_dir(outdir)
    device = cfg.device

    ds = ColoredMNIST(root=os.path.join(outdir, "data_cache"), train=(split == "train"))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)

    T = cfg.T
    t_grid = torch.linspace(0, T - 1, steps=num_t_points, dtype=torch.long)
    alpha_bar = ddpm.params["alpha_bar"].to(device)
    snr_all = snr_from_alpha_bar(alpha_bar)

    eps_mse_list: List[float] = []
    x0_mse_list: List[float] = []
    x0_psnr_list: List[float] = []
    w_eps_mse_list: List[float] = []
    w_x0_mse_list: List[float] = []

    for j, t_val in enumerate(t_grid):
        t_int = int(t_val.item())
        print(f"[{split}] t={t_int} ({j+1}/{len(t_grid)})")

        total_eps_mse = 0.0
        total_x0_mse = 0.0
        total_w_eps = 0.0
        total_w_x0 = 0.0
        total_samples = 0

        w_t = (1.0 / (1.0 + snr_all[t_int])).item()

        done = 0
        for x0, _ in dl:
            x0 = x0.to(device)
            B = x0.size(0)
            t = torch.full((B,), t_int, device=device, dtype=torch.long)

            noise = torch.randn_like(x0)
            x_t = ddpm.q_sample(x0, t, noise)

            eps_pred = model(x_t, t)
            x0_pred = ddpm.predict_x0_from_eps(x_t, t, eps_pred)

            eps_mse = torch.mean((eps_pred - noise) ** 2).item()
            x0_mse = torch.mean((x0_pred - x0) ** 2).item()

            total_eps_mse += eps_mse * B
            total_x0_mse += x0_mse * B
            total_w_eps += (w_t * eps_mse) * B
            total_w_x0 += (w_t * x0_mse) * B
            total_samples += B

            done += 1
            if done >= max_batches:
                break

        avg_eps = total_eps_mse / total_samples
        avg_x0 = total_x0_mse / total_samples
        avg_w_eps = total_w_eps / total_samples
        avg_w_x0 = total_w_x0 / total_samples

        eps_mse_list.append(avg_eps)
        x0_mse_list.append(avg_x0)
        w_eps_mse_list.append(avg_w_eps)
        w_x0_mse_list.append(avg_w_x0)

        # PSNR по среднему MSE (аппрокс)
        x0_psnr_list.append(float(psnr_from_mse(torch.tensor(avg_x0), data_range=2.0).item()))

    t_np = t_grid.numpy()

    def save_line(y, fname, ylabel, title):
        plt.figure()
        plt.plot(t_np, y, marker="o")
        plt.xlabel("t")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, fname))
        plt.close()

    save_line(eps_mse_list, "eps_mse_vs_t.png", "MSE(eps)", f"{split}: MSE(eps_pred, eps_true) vs t")
    save_line(x0_mse_list, "x0_mse_vs_t.png", "MSE(x0)", f"{split}: MSE(x0_pred, x0_true) vs t")
    save_line(x0_psnr_list, "x0_psnr_vs_t.png", "PSNR (dB)", f"{split}: PSNR(x0_pred, x0_true) vs t")
    save_line(w_eps_mse_list, "w_eps_mse_vs_t.png", "Weighted MSE(eps)", f"{split}: weighted MSE(eps) vs t")
    save_line(w_x0_mse_list, "w_x0_mse_vs_t.png", "Weighted MSE(x0)", f"{split}: weighted MSE(x0) vs t")

    data = {
        "t": t_np.tolist(),
        "eps_mse": eps_mse_list,
        "x0_mse": x0_mse_list,
        "x0_psnr": x0_psnr_list,
        "w_eps_mse": w_eps_mse_list,
        "w_x0_mse": w_x0_mse_list,
    }
    with open(os.path.join(outdir, "error_vs_t.json"), "w") as f:
        json.dump(data, f, indent=2)

    return data


# =========================
#  Forward early-stop methodology (teacher req #4)
# =========================

@torch.no_grad()
def forward_early_stop_reconstruction(cfg: DiffusionConfig,
                                      ddpm: DDPM,
                                      model: nn.Module,
                                      outdir: str,
                                      t_stop: int,
                                      num_steps: int,
                                      batch_size: int,
                                      max_batches: int,
                                      num_workers: int = 0) -> Dict[str, float]:
    """
    Вариант "early-stop for forward process":
      - берем реальные x0 (test)
      - делаем forward до t_stop: x_t = q(x0, t_stop)
      - пытаемся восстановить x0, используя укороченный reverse (DDIM-ODE) на сетке t_stop -> 0

    Метрики:
      - reconstruction MSE/PSNR на test
      - сохраняем примеры (x0, x_t, x_rec)
    """
    outdir = ensure_dir(outdir)
    device = cfg.device
    t_stop = int(min(max(t_stop, 1), cfg.T - 1))

    ds = ColoredMNIST(root=os.path.join(outdir, "data_cache"), train=False)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)

    # сетка шагов: t_stop -> 0
    timesteps = torch.linspace(t_stop, 0, num_steps, dtype=torch.long, device=device)
    timesteps = torch.unique(timesteps, sorted=True)
    timesteps = list(reversed(timesteps.tolist()))

    alpha_bar = ddpm.params["alpha_bar"].to(device)

    total_mse = 0.0
    total_psnr = 0.0
    total_samples = 0

    saved = 0
    example_dir = ensure_dir(os.path.join(outdir, "examples"))

    for b, (x0, _) in enumerate(dl):
        x0 = x0.to(device)

        # forward до t_stop
        B = x0.size(0)
        t0 = torch.full((B,), t_stop, device=device, dtype=torch.long)
        noise = torch.randn_like(x0)
        x_t = ddpm.q_sample(x0, t0, noise)

        # reverse укороченный (DDIM-ODE)
        x = x_t
        for i, t_int in enumerate(timesteps):
            t_int = int(t_int)
            t = torch.full((B,), t_int, device=device, dtype=torch.long)
            eps = model(x, t)

            ab_t = alpha_bar[t_int]
            sqrt_ab_t = torch.sqrt(ab_t)
            sqrt_om_t = torch.sqrt(1.0 - ab_t)
            x0_pred = (x - sqrt_om_t * eps) / (sqrt_ab_t + 1e-12)

            if i == len(timesteps) - 1:
                x = x0_pred
                break

            t_next = int(timesteps[i + 1])
            ab_next = alpha_bar[t_next]
            sqrt_ab_next = torch.sqrt(ab_next)
            sqrt_om_next = torch.sqrt(1.0 - ab_next)
            x = sqrt_ab_next * x0_pred + sqrt_om_next * eps

        x_rec = x

        mse = torch.mean((x_rec - x0) ** 2, dim=[1,2,3])  # [B]
        psnr = psnr_from_mse(mse, data_range=2.0)          # [B]

        total_mse += float(mse.mean().item()) * B
        total_psnr += float(psnr.mean().item()) * B
        total_samples += B

        # сохраняем несколько примеров
        if saved < 2:
            # сетка: x0, x_t, x_rec
            grid = torch.cat([x0[:16], x_t[:16], x_rec[:16]], dim=0)
            save_image(denorm(grid), os.path.join(example_dir, f"recon_t{t_stop}_steps{num_steps}_batch{saved}.png"), nrow=16)
            saved += 1

        if b + 1 >= max_batches:
            break

    avg_mse = total_mse / total_samples
    avg_psnr = total_psnr / total_samples

    result = {
        "t_stop": int(t_stop),
        "num_steps": int(num_steps),
        "avg_recon_mse": float(avg_mse),
        "avg_recon_psnr_db": float(avg_psnr),
        "samples": int(total_samples),
    }
    with open(os.path.join(outdir, "forward_early_stop_recon.json"), "w") as f:
        json.dump(result, f, indent=2)

    return result


# =========================
#  Benchmark: speed + speedups + sample grids + reports
# =========================

def _bench_one(name: str,
               fn,
               fn_kwargs: Dict,
               device: str,
               outdir: str,
               warmup: int,
               repeats: int) -> Dict[str, float]:
    """
    Возвращает:
      - time_mean
      - time_std
      - time_min
    """
    times: List[float] = []

    # warmup
    for _ in range(warmup):
        cuda_sync_if_needed(device)
        _ = fn(**fn_kwargs)
        cuda_sync_if_needed(device)

    # repeats
    for _ in range(repeats):
        cuda_sync_if_needed(device)
        t0 = time.perf_counter()
        x = fn(**fn_kwargs)
        cuda_sync_if_needed(device)
        t1 = time.perf_counter()
        dt = t1 - t0
        times.append(dt)

    # save one sample image (последний x)
    save_image(denorm(x), os.path.join(outdir, "samples", f"{name}.png"), nrow=8)

    t_mean = float(sum(times) / len(times))
    t_min = float(min(times))
    # std
    var = sum((t - t_mean) ** 2 for t in times) / max(len(times) - 1, 1)
    t_std = float(math.sqrt(var))

    return {"time_mean": t_mean, "time_std": t_std, "time_min": t_min}


def run_speed_benchmark(cfg: DiffusionConfig,
                        ddpm: DDPM,
                        model: nn.Module,
                        outdir: str,
                        batch_size: int,
                        steps50: int,
                        es_tstop: int,
                        hsivi_K: int,
                        hsivi_scale: float,
                        warmup: int,
                        repeats: int) -> Dict[str, Dict[str, float]]:
    outdir = ensure_dir(outdir)
    ensure_dir(os.path.join(outdir, "samples"))
    ensure_dir(os.path.join(outdir, "plots"))

    device = cfg.device

    methods = {
        "ddpm_full": (sample_ddpm_full, {"model": model, "ddpm": ddpm, "batch_size": batch_size}),
        "ddpm_coarse50": (sample_ddpm_coarse, {"model": model, "ddpm": ddpm, "batch_size": batch_size, "num_steps": steps50}),
        "ddim50": (sample_ddim, {"model": model, "ddpm": ddpm, "batch_size": batch_size, "num_steps": steps50, "eta": 0.0}),
        "explint50": (sample_explint, {"model": model, "ddpm": ddpm, "batch_size": batch_size, "num_steps": steps50}),
        "deis_like50": (sample_deis_like, {"model": model, "ddpm": ddpm, "batch_size": batch_size, "num_steps": steps50}),
        "es_ddpm": (sample_es_ddpm, {"model": model, "ddpm": ddpm, "batch_size": batch_size, "t_stop": es_tstop, "num_steps": steps50}),
        "hsivi50": (sample_hsivi, {"model": model, "ddpm": ddpm, "batch_size": batch_size, "num_steps": steps50, "K": hsivi_K, "perturb_scale": hsivi_scale}),
    }

    results: Dict[str, Dict[str, float]] = {}

    print("\n=== SPEED BENCHMARK ===")
    for name, (fn, kwargs) in methods.items():
        print(f"\n[{name}]")
        res = _bench_one(name, fn, kwargs, device, outdir, warmup=warmup, repeats=repeats)
        print(f"time_mean={res['time_mean']:.4f}s  time_std={res['time_std']:.4f}s  time_min={res['time_min']:.4f}s")
        results[name] = res

    # speedups vs ddpm_full (use mean)
    base = results["ddpm_full"]["time_mean"]
    speedups = {k: (base / v["time_mean"]) for k, v in results.items()}

    # save JSON
    with open(os.path.join(outdir, "times.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(outdir, "speedups.json"), "w") as f:
        json.dump({k: float(v) for k, v in speedups.items()}, f, indent=2)

    # plots
    names = list(results.keys())
    times = [results[n]["time_mean"] for n in names]
    su = [speedups[n] for n in names]

    plt.figure()
    plt.bar(names, times)
    plt.ylabel("time (s), mean")
    plt.title("Sampling time per method")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "plots", "times_bar.png"))
    plt.close()

    plt.figure()
    plt.bar(names, su)
    plt.ylabel("speedup vs ddpm_full (x)")
    plt.title("Speedups per method")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "plots", "speedups_bar.png"))
    plt.close()

    # CSV summary
    with open(os.path.join(outdir, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "time_mean_s", "time_std_s", "time_min_s", "speedup_vs_ddpm_full"])
        for n in names:
            w.writerow([n, results[n]["time_mean"], results[n]["time_std"], results[n]["time_min"], speedups[n]])

    print("\nSpeedups vs ddpm_full:")
    for n in names:
        print(f"{n:12s} : {speedups[n]:.2f}x")

    return {
        "times": results,
        "speedups": {k: float(v) for k, v in speedups.items()}
    }


# =========================
#  Extra: quality proxy on generated samples (simple stats)
# =========================

@torch.no_grad()
def generated_sample_stats(x: torch.Tensor) -> Dict[str, float]:
    # простые, дешевые метрики "для красоты": mean/std/min/max в пикселях
    return {
        "mean": float(x.mean().item()),
        "std": float(x.std().item()),
        "min": float(x.min().item()),
        "max": float(x.max().item()),
    }


@torch.no_grad()
def compare_generated_stats(cfg: DiffusionConfig,
                            ddpm: DDPM,
                            model: nn.Module,
                            outdir: str,
                            batch_size: int,
                            steps50: int,
                            es_tstop: int,
                            hsivi_K: int,
                            hsivi_scale: float) -> None:
    """
    Не заменяет FID/IS, но даёт нормальный "отчёт" по распределениям
    (и удобно для презентации).
    """
    outdir = ensure_dir(outdir)
    methods = {
        "ddim50": lambda: sample_ddim(model, ddpm, batch_size=batch_size, num_steps=steps50, eta=0.0),
        "explint50": lambda: sample_explint(model, ddpm, batch_size=batch_size, num_steps=steps50),
        "deis_like50": lambda: sample_deis_like(model, ddpm, batch_size=batch_size, num_steps=steps50),
        "es_ddpm": lambda: sample_es_ddpm(model, ddpm, batch_size=batch_size, t_stop=es_tstop, num_steps=steps50),
        "hsivi50": lambda: sample_hsivi(model, ddpm, batch_size=batch_size, num_steps=steps50, K=hsivi_K, perturb_scale=hsivi_scale),
    }

    report = {}
    for name, fn in methods.items():
        x = fn()
        report[name] = generated_sample_stats(x)

    with open(os.path.join(outdir, "generated_stats.json"), "w") as f:
        json.dump(report, f, indent=2)

    # plot std bars
    names = list(report.keys())
    stds = [report[n]["std"] for n in names]
    means = [report[n]["mean"] for n in names]

    plt.figure()
    plt.bar(names, stds)
    plt.ylabel("pixel std")
    plt.title("Generated sample pixel std (proxy diversity)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "gen_std_bar.png"))
    plt.close()

    plt.figure()
    plt.bar(names, means)
    plt.ylabel("pixel mean")
    plt.title("Generated sample pixel mean")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "gen_mean_bar.png"))
    plt.close()


# =========================
#  Full evaluation pipeline (teacher tasks 1–5)
# =========================

def run_full_eval(cfg: DiffusionConfig,
                  ckpt_path: str,
                  outdir: str,
                  batch_size_bench: int,
                  batch_size_err: int,
                  steps50: int,
                  es_tstop: int,
                  hsivi_K: int,
                  hsivi_scale: float,
                  err_num_t_points: int,
                  err_max_batches: int,
                  recon_max_batches: int,
                  warmup: int,
                  repeats: int,
                  num_workers: int = 0) -> str:
    set_seed(cfg.seed)
    run_dir = ensure_dir(os.path.join(outdir, f"eval_{now_run_id()}"))

    # subfolders (не перемешиваются)
    snr_dir = ensure_dir(os.path.join(run_dir, "snr"))
    speed_dir = ensure_dir(os.path.join(run_dir, "speed"))
    daed_train_dir = ensure_dir(os.path.join(run_dir, "daed_train"))
    daed_test_dir = ensure_dir(os.path.join(run_dir, "daed_test"))
    recon_dir = ensure_dir(os.path.join(run_dir, "forward_early_stop"))
    genstat_dir = ensure_dir(os.path.join(run_dir, "generated_stats"))

    ddpm, model = load_model(cfg, ckpt_path)

    # 1) SNR study
    analyze_snr(ddpm, snr_dir)

    # 2) speed + acceleration compare
    speed_report = run_speed_benchmark(
        cfg, ddpm, model,
        outdir=speed_dir,
        batch_size=batch_size_bench,
        steps50=steps50,
        es_tstop=es_tstop,
        hsivi_K=hsivi_K,
        hsivi_scale=hsivi_scale,
        warmup=warmup,
        repeats=repeats,
    )

    # 3) DAED-like: error vs t (train + test)
    _ = eval_error_vs_t(
        cfg, ddpm, model,
        outdir=daed_train_dir,
        split="train",
        num_t_points=err_num_t_points,
        max_batches=err_max_batches,
        batch_size=batch_size_err,
        num_workers=num_workers,
    )
    _ = eval_error_vs_t(
        cfg, ddpm, model,
        outdir=daed_test_dir,
        split="test",
        num_t_points=err_num_t_points,
        max_batches=err_max_batches,
        batch_size=batch_size_err,
        num_workers=num_workers,
    )

    # 4) forward early-stop methodology
    recon_report = forward_early_stop_reconstruction(
        cfg, ddpm, model,
        outdir=recon_dir,
        t_stop=es_tstop,
        num_steps=steps50,
        batch_size=batch_size_err,
        max_batches=recon_max_batches,
        num_workers=num_workers,
    )

    # 5) extra: generated stats compare (для красоты/отчёта)
    compare_generated_stats(
        cfg, ddpm, model,
        outdir=genstat_dir,
        batch_size=batch_size_bench,
        steps50=steps50,
        es_tstop=es_tstop,
        hsivi_K=hsivi_K,
        hsivi_scale=hsivi_scale,
    )

    # overall report
    report = {
        "ckpt": ckpt_path,
        "cfg": cfg.__dict__,
        "speed": speed_report,
        "forward_early_stop_recon": recon_report,
        "params": {
            "batch_size_bench": batch_size_bench,
            "batch_size_err": batch_size_err,
            "steps50": steps50,
            "es_tstop": es_tstop,
            "hsivi_K": hsivi_K,
            "hsivi_scale": hsivi_scale,
            "err_num_t_points": err_num_t_points,
            "err_max_batches": err_max_batches,
            "recon_max_batches": recon_max_batches,
            "warmup": warmup,
            "repeats": repeats,
        }
    }
    with open(os.path.join(run_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print("\nFULL EVAL DONE:", run_dir)
    return run_dir


# =========================
#  CLI (command-line interface)
# =========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="eval",
                        choices=["train", "bench", "analyze", "eval"],
                        help="train: обучить; bench: только скорость; analyze: только SNR/DAED/forward-early-stop; eval: всё вместе")
    parser.add_argument("--outdir", type=str, default="runs")

    parser.add_argument("--schedule", type=str, default="cosine", choices=["cosine", "linear"])
    parser.add_argument("--T", type=int, default=1000)

    # train args
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--use_snr_weighted", action="store_true")

    # ckpt for eval/bench/analyze
    parser.add_argument("--ckpt", type=str, default="", help="path to .pth checkpoint")

    # eval/bench args
    parser.add_argument("--batch_size_bench", type=int, default=64)
    parser.add_argument("--batch_size_err", type=int, default=128)

    parser.add_argument("--steps50", type=int, default=50)
    parser.add_argument("--es_tstop", type=int, default=400)

    parser.add_argument("--hsivi_K", type=int, default=4)
    parser.add_argument("--hsivi_scale", type=float, default=0.15)

    parser.add_argument("--err_num_t_points", type=int, default=20)
    parser.add_argument("--err_max_batches", type=int, default=50)
    parser.add_argument("--recon_max_batches", type=int, default=20)

    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)

    args = parser.parse_args()

    cfg = DiffusionConfig(
        schedule=args.schedule,
        T=args.T,
        seed=args.seed,
    )

    ensure_dir(args.outdir)

    if args.mode == "train":
        train(
            cfg=cfg,
            outdir=args.outdir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            use_snr_weighted=bool(args.use_snr_weighted),
            num_workers=args.num_workers,
        )
        return

    if args.ckpt == "":
        raise ValueError("Для mode=bench/analyze/eval нужно указать --ckpt путь к .pth")

    # load once
    set_seed(cfg.seed)
    ddpm, model = load_model(cfg, args.ckpt)

    run_dir = ensure_dir(os.path.join(args.outdir, f"{args.mode}_{now_run_id()}"))
    with open(os.path.join(run_dir, "meta.json"), "w") as f:
        json.dump({
            "mode": args.mode,
            "ckpt": args.ckpt,
            "cfg": cfg.__dict__,
            "args": vars(args),
        }, f, indent=2)

    if args.mode == "bench":
        _ = run_speed_benchmark(
            cfg, ddpm, model,
            outdir=ensure_dir(os.path.join(run_dir, "speed")),
            batch_size=args.batch_size_bench,
            steps50=args.steps50,
            es_tstop=args.es_tstop,
            hsivi_K=args.hsivi_K,
            hsivi_scale=args.hsivi_scale,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        compare_generated_stats(
            cfg, ddpm, model,
            outdir=ensure_dir(os.path.join(run_dir, "generated_stats")),
            batch_size=args.batch_size_bench,
            steps50=args.steps50,
            es_tstop=args.es_tstop,
            hsivi_K=args.hsivi_K,
            hsivi_scale=args.hsivi_scale,
        )
        print("BENCH DONE:", run_dir)
        return

    if args.mode == "analyze":
        analyze_snr(ddpm, ensure_dir(os.path.join(run_dir, "snr")))
        _ = eval_error_vs_t(
            cfg, ddpm, model,
            outdir=ensure_dir(os.path.join(run_dir, "daed_test")),
            split="test",
            num_t_points=args.err_num_t_points,
            max_batches=args.err_max_batches,
            batch_size=args.batch_size_err,
            num_workers=args.num_workers,
        )
        _ = forward_early_stop_reconstruction(
            cfg, ddpm, model,
            outdir=ensure_dir(os.path.join(run_dir, "forward_early_stop")),
            t_stop=args.es_tstop,
            num_steps=args.steps50,
            batch_size=args.batch_size_err,
            max_batches=args.recon_max_batches,
            num_workers=args.num_workers,
        )
        print("ANALYZE DONE:", run_dir)
        return

    # eval = everything
    _ = run_full_eval(
        cfg=cfg,
        ckpt_path=args.ckpt,
        outdir=args.outdir,
        batch_size_bench=args.batch_size_bench,
        batch_size_err=args.batch_size_err,
        steps50=args.steps50,
        es_tstop=args.es_tstop,
        hsivi_K=args.hsivi_K,
        hsivi_scale=args.hsivi_scale,
        err_num_t_points=args.err_num_t_points,
        err_max_batches=args.err_max_batches,
        recon_max_batches=args.recon_max_batches,
        warmup=args.warmup,
        repeats=args.repeats,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()