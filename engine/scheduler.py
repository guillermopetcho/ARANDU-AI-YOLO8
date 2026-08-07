import torch

import math

def build_scheduler(opt, w_steps, t_steps, final_lr_ratio=0.0, 
                    restarts=0, restart_decay=0.8):
    """Construye un scheduler LambdaLR con warmup lineal + decaimiento cosenoidal y restarts opcionales.
    
    Args:
        opt: Optimizer de PyTorch.
        w_steps: Pasos de warmup (rampa lineal de 10% a 100% del LR base).
        t_steps: Pasos totales de entrenamiento.
        final_lr_ratio: Fracción mínima del LR a la que decaerá al final.
        restarts: Número de restarts coseno después del warmup. 0 = monotónico.
        restart_decay: Factor de decaimiento del LR máximo en cada restart.
    """
    if restarts <= 0:
        def lr_lambda(step):
            if step < w_steps:
                return 0.10 + 0.90 * (step / max(1, w_steps))
            else:
                progress = (step - w_steps) / max(1, t_steps - w_steps)
                decay = 0.5 * (1.0 + math.cos(math.pi * progress))
                return final_lr_ratio + (1.0 - final_lr_ratio) * decay
    else:
        cycle_steps = (t_steps - w_steps) // (restarts + 1)
        def lr_lambda(step):
            if step < w_steps:
                return 0.10 + 0.90 * (step / max(1, w_steps))
            else:
                post_warmup = step - w_steps
                cycle = post_warmup // max(1, cycle_steps)
                pos_in_cycle = post_warmup % max(1, cycle_steps)
                cycle_progress = pos_in_cycle / max(1, cycle_steps)
                peak = restart_decay ** cycle
                decay = 0.5 * (1.0 + math.cos(math.pi * cycle_progress))
                return final_lr_ratio + (peak - final_lr_ratio) * decay
            
    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

@torch.no_grad()
def momentum_update(model_q, model_k, m):
    # Desenvuelver DDP y torch.compile para acceder a los parámetros reales
    # DDP: model_q.module; torch.compile: model_q._orig_mod
    q = model_q
    if hasattr(q, 'module'):       # DDP wrapper
        q = q.module
    if hasattr(q, '_orig_mod'):    # torch.compile wrapper
        q = q._orig_mod
    for param_q, param_k in zip(q.parameters(), model_k.parameters()):
        param_k.data = param_k.data * m + param_q.data * (1. - m)