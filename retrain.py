import os
import glob
import yaml
import subprocess
import sys


def _get_base_dir():
    """Retorna el directorio del script, independientemente del CWD.
    
    N11 FIX: Usar __file__ en vez de una ruta relativa al CWD para que el
    script funcione correctamente sin importar desde dónde se llame.
    Ej: `python /kaggle/working/ARANDU/retrain.py` desde /kaggle/working
    encontrará `config/moco.yaml` relativo al script, no al CWD.
    """
    return os.path.dirname(os.path.abspath(__file__))


def clean_workspace(base_dir):
    print("\n" + "="*50)
    print(" PREPARANDO RE-ENTRENAMIENTO DESDE CERO")
    print("="*50)

    config_path = os.path.join(base_dir, "config", "moco.yaml")
    if not os.path.exists(config_path):
        print(f" Error: No se encontró el archivo de configuración {config_path}")
        sys.exit(1)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    paths = config.get("paths", {})

    # Extraer las rutas del config para saber exactamente qué borrar
    ckpt_path    = paths.get("checkpoint_path",      "/kaggle/working/moco_checkpoint.pth")
    metrics_path = paths.get("metrics_path",         "/kaggle/working/ssl_metrics.json")
    encoder_path = paths.get("encoder_export_path",  "/kaggle/working/moco_encoder_ready.pth")

    # Lista explícita de archivos a eliminar (checkpoints, exportados y logs)
    to_delete = [
        ckpt_path,
        paths.get("best_checkpoint_path", "/kaggle/working/moco_best_checkpoint.pth"),
        ckpt_path.replace(".pth", "_best_geom.pth"),

        encoder_path,
        encoder_path.replace(".pth", "_head.pth"),
        encoder_path.replace(".pth", "_class_names.json"),

        metrics_path,
        metrics_path.replace(".json", "_log.csv"),
        metrics_path.replace(".json", "_projector.csv"),

        paths.get("index_cache_path", "/kaggle/working/train_index.npy"),
    ]

    # Limpiar bases de datos KNN cacheadas (hash del data_yaml en el nombre)
    base_cache = (
        "/kaggle/working/reference_db"
        if "kaggle" in paths.get("eval_train_root", "")
        else os.path.join(base_dir, "reference_db")
    )
    to_delete.extend(glob.glob(f"{base_cache}*.pt"))
    # También buscar en CWD por si alguna corrida anterior los dejó allí
    to_delete.extend(glob.glob("reference_db*.pt"))

    # Temporales de guardado atómico
    to_delete.extend(glob.glob(ckpt_path + "*.tmp"))

    deleted_count = 0
    for file_path in set(to_delete):
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"  🗑️  Eliminado: {file_path}")
                deleted_count += 1
            except Exception as e:
                print(f"    No se pudo eliminar {file_path}: {e}")

    if deleted_count == 0:
        print("   El entorno ya está limpio. No había checkpoints anteriores.")
    else:
        print(f"   Limpieza completada. {deleted_count} archivos eliminados.")


def _build_train_command(base_dir):
    """Construye el comando correcto para lanzar train.py.

    N3 FIX: train.py NO usa argparse — lee todo desde moco.yaml.
    Por eso NO se deben pasar sys.argv[1:] al subprocess.

    Casos soportados:
    1. Ejecución normal (1 GPU / CPU):
       `python retrain.py`  →  lanza `python train.py`

    2. Ejecución multi-GPU con torchrun (recomendado):
       `torchrun --nproc_per_node=2 retrain.py`
       → torchrun inyecta RANK, WORLD_SIZE, LOCAL_RANK en el env.
       → retrain.py detecta estas variables y re-lanza train.py
         también con torchrun para que los workers sean procesos DDP reales.

    3. NO soportado (causa el bug N3):
       `python retrain.py --nproc_per_node=2`  → train.py recibe args inválidos.
    """
    train_script = os.path.join(base_dir, "train.py")

    # Detectar si el caller fue torchrun (inyecta estas variables de entorno)
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    nproc = int(os.environ.get("LOCAL_WORLD_SIZE", world_size))

    if world_size > 1:
        # Modo DDP: re-lanzar train.py con torchrun usando los mismos parámetros
        # que torchrun ya configuró para retrain.py.
        # NOTA: No pasamos --rdzv_* porque torchrun ya estableció esas variables de entorno;
        # el nuevo torchrun las hereda del entorno del proceso actual.
        return ["torchrun", f"--nproc_per_node={nproc}", train_script]
    else:
        return [sys.executable, train_script]


def main():
    base_dir = _get_base_dir()
    clean_workspace(base_dir)

    print("\n INICIANDO NUEVO ENTRENAMIENTO...")
    print("-" * 50)

    cmd = _build_train_command(base_dir)
    print(f"   Comando: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n El entrenamiento finalizó con errores (código {e.returncode}).")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\n  Entrenamiento interrumpido por el usuario.")
        sys.exit(0)


if __name__ == "__main__":
    main()
