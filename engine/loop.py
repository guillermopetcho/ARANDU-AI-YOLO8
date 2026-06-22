"""engine/loop.py — Lógica del bucle de entrenamiento principal.

Extraído de train.py para simplificar el flujo principal y delegar
la lógica de epochs, evaluación, y rollback.
"""

import os
import csv
import logging
import warnings
import numpy as np
import torch
import torch.distributed as dist

from engine.checkpoint import (
    build_checkpoint_dict, save_checkpoint, load_weights_for_rollback
)
from engine.setup import make_eval_subset_loader
from evaluation.knn import extract_features_fast, fast_knn
from engine.controller import Action
from utils.metrics import get_module_stats


def get_model_module(model, is_distributed):
    return model.module if is_distributed else model


def handle_evaluation(
    epoch, model_q, eval_train_loader, eval_val_loader,
    device, is_distributed, CONFIG, logger, controller, metrics
):
    """Ejecuta la evaluación KNN y el análisis del espacio latente."""
    curr_acc = -1.0  # BUG-EMPTY-EVAL FIX: valor por defecto seguro si los arrays quedan vacíos
    eval_model = get_model_module(model_q, is_distributed)
    eval_model.eval()
    X_t, y_t = extract_features_fast(eval_model, eval_train_loader, device)
    X_v, y_v = extract_features_fast(eval_model, eval_val_loader, device)
    eval_model.train()

    # Guard: si alguno de los splits está vacío, el KNN y el SVD no tienen sentido.
    # Retornar CONTINUE sin modificar el controller para no disparar rollbacks falsos.
    if len(X_t) == 0 or len(X_v) == 0:
        logger.warning("⚠️ handle_evaluation: features vacías (loader vacío). Saltando eval.")
        return Action.CONTINUE, curr_acc

    curr_acc = fast_knn(X_t, y_t, X_v, y_v, k=CONFIG["eval"]["knn_k"])
    logger.info(f"KNN ACC: {curr_acc:.4f}")

    # C3 FIX: svdvals requiere al menos 2 muestras.
    SVD_MAX_SAMPLES = 2000
    # Bug #5 FIX: torch.from_numpy() zero-copy vs torch.tensor() que siempre copia.
    X_v_t = torch.from_numpy(np.ascontiguousarray(X_v, dtype=np.float32))
    if len(X_v_t) < 2:
        logger.warning(f"⚠️ eval_val_loader tiene solo {len(X_v_t)} muestras — SVD omitido.")
        embed_dim = CONFIG['moco'].get('dim', 512)
        # BUG-C4 FIX: X_v_t puede ser 1D (shape (0,) o (1,)) si extract_features_fast
        # devuelve el array de emergencia o una sola muestra. Acceder a shape[1] en un
        # tensor 1D lanza IndexError. Verificamos ndim antes de acceder a la dimensión.
        if X_v_t.ndim >= 2 and X_v_t.shape[0] > 0:
            mu_dim = X_v_t.shape[1]
        else:
            mu_dim = embed_dim
        metrics['mu'] = torch.zeros(mu_dim)
        metrics['eff_rank'] = 1.0
        return controller.step_epoch(epoch, curr_acc, metrics), curr_acc

    if len(X_v_t) > SVD_MAX_SAMPLES:
        perm = torch.randperm(len(X_v_t))[:SVD_MAX_SAMPLES]
        X_svd = X_v_t[perm]
    else:
        X_svd = X_v_t

    mu = X_svd.mean(dim=0)
    X_centered = X_svd - mu.unsqueeze(0)
    s = torch.linalg.svdvals(X_centered)
    p = (s**2) / ((s**2).sum() + 1e-8)
    p = torch.clamp(p, min=1e-6)
    p = p / p.sum()
    eff_rank = torch.exp(-(p * torch.log(p)).sum()).item()

    metrics['mu'] = mu
    metrics['eff_rank'] = eff_rank

    return controller.step_epoch(epoch, curr_acc, metrics), curr_acc


def handle_rollback(
    CONFIG, rank, use_wandb, global_step, model_q, model_k,
    optimizer, scaler, queue, is_compiled, is_distributed,
    warmup_steps, total_steps, final_lr_ratio, build_scheduler, trainer, controller, logger
):
    """Ejecuta la lógica de rollback a un checkpoint previo.
    
    INVARIANTE CLAVE: retorna el mismo global_step recibido (nunca retrocede).
    El step del checkpoint se usa internamente solo para el fast-forward del scheduler,
    pero el step de wandb sigue siendo monótono.
    """
    if rank == 0: logger.info("🔄 Iniciando proceso de Rollback...")
    if rank == 0 and use_wandb:
        try:
            import wandb
            wandb.log({"event/rollback": 1}, step=global_step)
        except Exception:
            pass

    rollback_ckpt_path = CONFIG["paths"]["best_checkpoint_path"]
    if not os.path.exists(rollback_ckpt_path):
        rollback_ckpt_path = CONFIG["paths"].get("checkpoint_path", "")
    if not rollback_ckpt_path or not os.path.exists(rollback_ckpt_path):
        if rank == 0: logger.warning("⚠️ Rollback solicitado pero no hay checkpoint. Continuando.")
        return global_step  # Sin cambios

    # ckpt_step se usa SOLO para el fast-forward del scheduler.
    # NO se retorna al caller para no romper la monotonícidad de wandb.
    ckpt_step = load_weights_for_rollback(
        path=rollback_ckpt_path,
        model_q=model_q, model_k=model_k,
        optimizer=optimizer, scaler=scaler, queue=queue,
        is_compiled=is_compiled, is_distributed=is_distributed,
    )

    # HIGH-2 FIX: El bug anterior aplicaba lr = initial_lr * 0.5 ANTES del fast-forward,
    # pero build_scheduler crea un LambdaLR que calcula lr = initial_lr * lambda(step),
    # sobrescribiendo la penalización en el primer scheduler.step().
    #
    # Solución: reconstruir el scheduler normalmente, hacer fast-forward, y DESPUÉS
    # reducir initial_lr al 50%. Esto garantiza que:
    #   1. El fast-forward posiciona el scheduler en el step correcto del checkpoint.
    #   2. La penalización 0.5x se aplica SOBRE el LR que el scheduler calculó.
    #   3. Futuros scheduler.step() usan el initial_lr reducido como base,
    #      preservando la penalización en el decaimiento cosenoidal.
    #   4. El floor de 1e-7 previene que rollbacks consecutivos colapsen el LR.
    scheduler = build_scheduler(optimizer, warmup_steps, total_steps, final_lr_ratio=final_lr_ratio)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(ckpt_step):  # Fast-forward al paso del checkpoint
            scheduler.step()

    # Aplicar penalización 0.5x DESPUÉS del fast-forward para que persista
    for param_group in optimizer.param_groups:
        param_group['initial_lr'] = max(param_group['initial_lr'] * 0.5, 1e-7)
        param_group['lr'] = max(param_group['lr'] * 0.5, 1e-7)

    trainer.scheduler = scheduler
    controller.warmup_aborted = True
    # CRIT-3 FIX: Resetear lr_scale tras rollback. El modelo fue restaurado a su mejor
    # versión, pero lr_scale podía estar en 0.25 (castigado por el PID). Sin este reset,
    # la próxima época aplica lr_step_factor derivado del lr_scale stale, causando un
    # doble castigo sobre un modelo recién restaurado.
    controller.lr_scale = 1.0
    controller.lr_step_factor = 1.0
    if rank == 0: logger.info("✅ Rollback completado. LR_scale reseteado a 1.0. Iniciando fase de Decaimiento Cosenoidal.")
    return global_step  # Retornar el step actual (monótono), NO el del checkpoint
