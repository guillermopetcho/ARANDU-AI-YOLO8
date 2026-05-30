import time
import logging
import torch
import torch.nn.functional as F
from torch.amp import autocast
import torch.distributed as dist
import contextlib
from collections import defaultdict
from tqdm.auto import tqdm

from utils.distributed import batch_shuffle_ddp, batch_unshuffle_ddp
from utils.metrics import compute_metrics
from engine.scheduler import momentum_update

class MoCoTrainer:
    def __init__(self, model_q, model_k, queue, optimizer, scheduler, scaler, config, device, is_distributed):
        self.model_q = model_q
        self.model_k = model_k
        self.queue = queue
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.config = config
        self.device = device
        self.is_distributed = is_distributed
        self.controller = config.get('_controller', None)
        self.last_unif = 0.0
        # E4 FIX: Resolver el tipo de device dinámicamente para que autocast funcione en CPU y GPU
        self.device_type = device.type if hasattr(device, 'type') else str(device).split(':')[0]
        # Peso de la pérdida de vistas locales (Multi-Crop)
        self.local_loss_weight = config['moco'].get('local_loss_weight', 0.5)

    def train_epoch(self, loader, epoch, global_step, total_steps, rank):
        self.model_q.train()
        epoch_loss, pos_sum, neg_sum, align_sum, unif_sum, std_sum = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        pos_sim_sum, neg_sim_sum, grad_norm_sum, grad_steps = 0.0, 0.0, 0.0, 0
        norm_sum, queue_std_sum = 0.0, 0.0
        global_loss_sum, local_loss_sum = 0.0, 0.0
        valid_steps = 0
        # R-5 FIX: Contador de batches válidos (no-NaN) dentro de la ventana de acumulación
        # actual. Necesario para calcular accum_count correctamente cuando hay batches NaN
        # saltados: `step` avanza aunque no se acumuló gradiente, causando denominador inflado.
        valid_in_window = 0
        # T1 FIX: Inicializar aliases antes del loop para evitar UnboundLocalError
        # si el primer batch produce NaN y el loop hace 'continue' sin asignarlos.
        q = k = l_pos = l_neg = None
        current_norm = 0.0  # B-NORM FIX: inicializar antes del loop para evitar UnboundLocalError en batch NaN

        pbar = tqdm(loader) if rank == 0 else loader
        epoch_start = time.time()

        for step, batch in enumerate(pbar):
            # F6 FIX: local_crops es [B, N, C, H, W] (5D).
            # channels_last solo aplica a tensores 4D — se aplica por slice dentro del loop.
            v_q, v_k, local_crops = batch
            # C-2 FIX: default_collate apila crops del mismo tamaño en tensor 5D [B, N, C, H, W].
            # Eso rompe el loop de shape_groups porque len() devuelve B (imágenes) en lugar de N (crops).
            # Normalizar a lista de tensores [B, C, H, W] para que el resto del pipeline sea correcto.
            if isinstance(local_crops, torch.Tensor) and local_crops.ndim == 5:
                local_crops = list(local_crops.unbind(dim=1))
            # local_crops es ahora una lista de tensores [ [B, C, H_i, W_i], ... ]
            
            # --- CURRICULUM DINÁMICO ---
            curriculum_epoch = self.config['training'].get('curriculum_epoch', 25)
            if epoch < curriculum_epoch and local_crops is not None:
                # Ignorar crops muy pequeños (64x64) antes de la época de estabilización
                local_crops = [crop for crop in local_crops if crop.shape[-1] >= 96]
                
            if len(local_crops) > 0:
                local_crops = [crop.to(self.device, non_blocking=True) for crop in local_crops]
            else:
                local_crops = None  # Placeholder vacío → desactivar multi-crop

            # --- Hiperparámetros dinámicos del Regulador Adaptativo ---
            if self.controller:
                momentum, temp = self.controller.get_dynamic_hyperparams(global_step, total_steps, self.last_unif)
            else:
                momentum, temp = 0.996, 0.2

            v_q = v_q.to(self.device, non_blocking=True, memory_format=torch.channels_last)
            v_k = v_k.to(self.device, non_blocking=True, memory_format=torch.channels_last)

            is_last_batch = (step + 1) == len(loader)
            is_accum_step = (step + 1) % self.config['training']['grad_accum_steps'] == 0
            is_accumulating = not (is_accum_step or is_last_batch)
            sync_context = self.model_q.no_sync() if (self.is_distributed and is_accumulating) else contextlib.nullcontext()

            with sync_context:
                with autocast(self.device_type, enabled=self.config['training']['use_amp']):
                    # === Vista Global 1: query usa predictor (MoCo v3) ===
                    q1, q1_norm = self.model_q(v_q, use_predictor=True, return_norm=True)
                    with torch.no_grad():
                        if self.is_distributed:
                            v_k_sh, idx1 = batch_shuffle_ddp(v_k)
                            k1 = self.model_k(v_k_sh)
                            k1 = batch_unshuffle_ddp(k1, idx1)
                        else:
                            k1 = self.model_k(v_k)  # key NO usa predictor

                    l_pos1 = torch.einsum('nc,nc->n', [q1, k1]).unsqueeze(-1)
                    l_neg1 = torch.einsum('nc,ck->nk', [q1, self.queue.queue.detach()])
                    logits1 = (torch.cat([l_pos1, l_neg1], dim=1) / temp).clamp(-15, 15)

                    # === Vista Global 2: simétrica ===
                    q2 = self.model_q(v_k, use_predictor=True)
                    with torch.no_grad():
                        if self.is_distributed:
                            v_q_sh, idx2 = batch_shuffle_ddp(v_q)
                            k2 = self.model_k(v_q_sh)
                            k2 = batch_unshuffle_ddp(k2, idx2)
                        else:
                            k2 = self.model_k(v_q)

                    l_pos2 = torch.einsum('nc,nc->n', [q2, k2]).unsqueeze(-1)
                    l_neg2 = torch.einsum('nc,ck->nk', [q2, self.queue.queue.detach()])
                    logits2 = (torch.cat([l_pos2, l_neg2], dim=1) / temp).clamp(-15, 15)

                    labels = torch.zeros(logits1.shape[0], dtype=torch.long, device=self.device)
                    loss_global = (F.cross_entropy(logits1, labels) + F.cross_entropy(logits2, labels)) * 0.5

                    # === Vistas Locales (Multi-Crop DINO-style) Vectorizadas ===
                    # T5 FIX: Vectorización completa para maximizar Throughput.
                    # Pasamos de O(N) llamadas secuenciales a O(1) llamada masiva.
                    #
                    # BUG-LOCAL FIX: Inicializar como tensor (no float 0.0) para que
                    # la suma 'tensor_loss + cross_entropy_tensor' preserve el grafo
                    # de cómputo bajo AMP. Con float 0.0 como acumulador, el primer
                    # sumando rompe el grafo y el gradiente NO se propaga a local_crops.
                    # C-3 FIX: torch.zeros([]) crea un escalar-cero sin autograd graph propio.
                    # Al sumarse con loss_global (que sí tiene grad), el resultado hereda el grafo
                    # correctamente. Usar zeros([]) en lugar de tensor(0.0) evita ambigüedad de
                    # dtype bajo AMP: el acumulador no fuerza un cast a float32 en la suma.
                    loss_local = torch.zeros([], device=self.device)
                    if local_crops is not None and len(local_crops) > 0:
                        # Agrupar crops por resolución para maximizar throughput en GPU
                        shape_groups = defaultdict(list)  # import movido al top del módulo
                        for crop in local_crops:
                            shape_groups[crop.shape[-1]].append(crop)
                            
                        for size, crops in shape_groups.items():
                            N_crops = len(crops)
                            # Concatenar todos los crops de esta resolución: [B*N_crops, C, H, W]
                            v_local = torch.cat(crops, dim=0).contiguous().to(memory_format=torch.channels_last)
                            q_local = self.model_q(v_local, use_predictor=True)
                            
                            # Expandir k1 y labels de [B, ...] a [B*N_crops, ...]
                            # torch.cat concatena secuencialmente, por lo que k1.repeat es el match correcto
                            k1_exp = k1.repeat(N_crops, 1)
                            labels_local = labels.repeat(N_crops)
                            
                            l_pos_l = torch.einsum('nc,nc->n', [q_local, k1_exp]).unsqueeze(-1)
                            l_neg_l = torch.einsum('nc,ck->nk', [q_local, self.queue.queue.detach()])
                            logits_l = (torch.cat([l_pos_l, l_neg_l], dim=1) / temp).clamp(-15, 15)
                            
                            # Acumular el loss multiplicando por N_crops para mantener escala
                            loss_local += F.cross_entropy(logits_l, labels_local) * N_crops
                            
                        loss_local = loss_local / len(local_crops)

                    loss = loss_global + self.local_loss_weight * loss_local

                    # Aliases para métricas
                    q, k, l_pos, l_neg = q1, k1, l_pos1, l_neg1
                    current_norm = q1_norm.item()

                # C3 FIX: usar .item() para evitar ambigüedad de torch.Tensor en contexto bool
                is_finite_val = loss.isfinite().item()
                is_finite = torch.tensor(1 if is_finite_val else 0, device=self.device)
                if self.is_distributed:
                    dist.all_reduce(is_finite, op=dist.ReduceOp.MIN)

                if is_finite.item() == 0:
                    self.optimizer.zero_grad(set_to_none=True)  # Limpiar grads contaminados
                    continue

                # R-5 FIX: Incrementar aquí, justo después de confirmar que el batch es válido.
                valid_in_window += 1

                accum_count = self.config['training']['grad_accum_steps']
                if is_last_batch and not is_accum_step:
                    # Último batch parcial: denominador real = batches válidos en esta ventana.
                    # R-5 FIX: usar valid_in_window en lugar de (step % grad_accum_steps) + 1.
                    # step cuenta todos los batches (incluyendo NaN saltados), por lo que
                    # el denominador anterior estaba inflado cuando había batches NaN.
                    accum_count = max(1, valid_in_window)
                self.scaler.scale(loss / accum_count).backward()

            # === Paso de Optimización ===
            if not is_accumulating:
                # B8 FIX: unscale_ primero en ambos paths para medir grad_norm sobre gradientes reales.
                if self.config['training']['use_amp']:
                    self.scaler.unscale_(self.optimizer)

                # L2 FIX: Medir grad_norm ANTES de clip_grad_norm_ para capturar
                # el valor real pre-clip. El código anterior medía post-clip, por lo
                # que nunca veia valores > 1.0 (siempre estaban cortados).
                # Se mide en todos los pasos de optimización para no perder spikes
                # que ocurran entre múltiplos de 50, usando avg del epoch como reporte.
                gn = sum(
                    p.grad.data.norm(2).item() ** 2
                    for p in self.model_q.parameters() if p.grad is not None
                ) ** 0.5
                grad_norm_sum += gn
                grad_steps += 1

                grad_clip_val = self.config['training'].get('grad_clip', 0.5)
                torch.nn.utils.clip_grad_norm_(self.model_q.parameters(), grad_clip_val)

                if self.config['training']['use_amp']:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.scheduler.step()


                
                self.optimizer.zero_grad(set_to_none=True)
                global_step += 1
                valid_in_window = 0  # R-5 FIX: reset para la siguiente ventana de acumulación

                with torch.no_grad():
                    momentum_update(self.model_q, self.model_k, momentum)
                    # BUG-ENQUEUE FIX: El enqueue de la queue DEBE estar dentro del
                    # bloque de optimización (if not is_accumulating). Antes estaba
                    # un nivel arriba, por lo que en pasos de acumulación se encolaba
                    # k del batch ANTERIOR (k ya asignado fuera del scope de control),
                    # contaminando la queue con representaciones desactualizadas.
                    #
                    # BUG-KNONE FIX: k puede ser None si TODOS los batches del
                    # primer ciclo de acumulación fueron NaN (continue en L155
                    # sin asignar k en L144). enqueue_dequeue(None) causa
                    # AttributeError en keys.shape. Guard obligatorio.
                    if k is not None:
                        self.queue.enqueue_dequeue(k, step=global_step)

            # === Métricas === (solo si q y k fueron asignados en este step)
            if q is not None:
                epoch_loss += loss.item()
                global_loss_sum += loss_global.item()
                local_loss_sum += loss_local.item() if isinstance(loss_local, torch.Tensor) and loss_local.numel() > 0 else 0.0
                valid_steps += 1  # L1 FIX: Solo contar batches procesados exitosamente
                with torch.no_grad():
                    metrics_step = compute_metrics(q, k)
                    pos_sum += l_pos.mean().item()
                    neg_sum += l_neg.mean().item()
                    align_sum += metrics_step['alignment']
                    unif_sum += metrics_step['uniformity']
                    pos_sim_sum += metrics_step['pos_sim']
                    neg_sim_sum += metrics_step['neg_sim']
                    std_sum += metrics_step['std']
                    norm_sum += current_norm
                    queue_std_sum += self.queue.queue.std().item()
                    self.last_unif = metrics_step['uniformity']

        # L1 FIX: Usar valid_steps (batches realmente procesados) en vez de len(loader)
        # para que los batches NaN saltados no diluyan las métricas reportadas.
        num_steps = max(1, valid_steps)

        if self.is_distributed:
            metrics_tensor = torch.tensor([
                epoch_loss, pos_sum, neg_sum, align_sum, unif_sum, std_sum,
                pos_sim_sum, neg_sim_sum, grad_norm_sum, float(grad_steps), float(valid_steps),
                norm_sum, queue_std_sum, global_loss_sum, local_loss_sum
            ], device=self.device, dtype=torch.float32)
            dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
            
            epoch_loss, pos_sum, neg_sum, align_sum, unif_sum, std_sum, \
                pos_sim_sum, neg_sim_sum, grad_norm_sum, grad_steps, valid_steps, \
                norm_sum, queue_std_sum, global_loss_sum, local_loss_sum = metrics_tensor.tolist()
            grad_steps = int(grad_steps)
            # Usar la suma total de pasos en todos los procesos como denominador
            num_steps = max(1, int(valid_steps))

        # I1 FIX: Monitorear integridad del dataset.
        # Se lee desde _load_errors.Value (compartido entre workers via multiprocessing)
        # en lugar del atributo de instancia clásico, que sería invisible entre procesos forkeados.
        load_errors = 0
        if hasattr(loader.dataset, '_load_errors'):
            with loader.dataset._load_errors.get_lock():
                load_errors = loader.dataset._load_errors.value
                loader.dataset._load_errors.value = 0  # Reset atómico para la siguiente época
        
        # M1 FIX: Usar len(loader.dataset) como denominador (total de imágenes reales
        # del dataset de este rank), en lugar de len(loader)*batch_size que asume
        # batches completos y es incorrecto para el último batch (sin drop_last en eval).
        total_samples = max(1, len(loader.dataset))
        error_rate = (load_errors / total_samples) * 100
        if error_rate > 1.0 and rank == 0:
            logging.getLogger("AranduSSL").warning(f"⚠️ Alta tasa de errores de carga: {error_rate:.2f}% ({load_errors} imágenes).")

        return {
            'loss': epoch_loss / num_steps,
            'global_loss': global_loss_sum / num_steps,
            'local_loss': local_loss_sum / num_steps,
            'pos': pos_sum / num_steps,
            'neg': neg_sum / num_steps,
            'margin': (pos_sum - neg_sum) / num_steps,
            'align': align_sum / num_steps,
            'unif': unif_sum / num_steps,
            'pos_sim': pos_sim_sum / num_steps,
            'neg_sim': neg_sim_sum / num_steps,
            'std': std_sum / num_steps,
            'norm': norm_sum / num_steps,
            'queue_std': queue_std_sum / num_steps,
            # M-2 FIX: float('nan') en lugar de None — csv.writer escribe None como el string "None"
            # que corrompe la columna gn para pandas. float('nan') se serializa como
            # vacío en CSV y WandB lo omite limpiamente sin romper gráficas.
            'gn': grad_norm_sum / grad_steps if grad_steps > 0 else float('nan'),
            # N1 FIX: el numerador correcto es imágenes REALMENTE procesadas por todos los ranks.
            # - valid_steps: batches procesados exitosamente en ESTE rank (ya sumado entre ranks en DDP)
            # - batch_size: imágenes por batch por rank
            # - El denominador `epoch_start` mide el tiempo real del epoch completo
            # Bug anterior: len(loader)*batch_size*world_size doblaba el conteo en DDP porque
            # len(loader) en un DistributedSampler ya es N_total/world_size, y multiplicar
            # luego por world_size da N_total correctamente, pero ignoraba batches NaN saltados.
            'tput': (num_steps * self.config['training']['batch_size']) / max(1e-6, time.time() - epoch_start),
            'data_err': error_rate
        }, global_step