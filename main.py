"""
main.py — Orquesta el pipeline completo.

Uso:
  python main.py                          # smoke test (bottle, 5 epochs, 128px)
  python main.py --full                   # entrenamiento baseline (todas las clases)
  python main.py --full --category wood   # solo una clase en full
  python main.py --rerun                  # re-entrena clases borderline con LATENT_DIM=64
  python main.py --eval-only --full       # re-evalua checkpoints con multi-scale SSIM
  python main.py --v2 --full             # re-entrena con perceptual loss + KL annealing
  python main.py --v2 --category bottle  # solo una clase con v2

Decisiones de diseño documentadas:
  - LATENT_DIM=128 (run original): punto de arranque acordado en el plan.
    Resultado: clases con objetos fijos (metal_nut, hazelnut, tile) funcionan bien.
    Clases con objetos de orientacion variable o textura fina (screw, grid, toothbrush)
    fallan estructuralmente.

  - LATENT_DIM=64 (--rerun): compresion mas agresiva. Resultado: igual o peor.
    capsule regresiona (pierde estructura bicolor). Descartado.

  - --v2: perceptual loss (VGG16) + KL annealing. Apunta a resolver
    el problema de blob gris en bottle, zipper, leather, wood.
    No resuelve fallos estructurales (screw, grid, toothbrush, carpet).
"""

import argparse
import json
import os
import torch
from pathlib import Path

from src.dataset  import get_dataloaders
from src.model    import ConvVAE
from src.train    import train
from src.utils    import PerceptualLoss
from src.evaluate import evaluate_category, save_figures, save_f1_curve

# ─────────────────────────────────────────────
# Configuración
# ─────────────────────────────────────────────

SMOKE = {
    "img_size":    128,
    "epochs":      5,
    "latent_dim":  128,
    "beta":        0.5,
    "lambda1":     0.5,
    "lambda2":     0.5,
    "lambda3":     0.0,
    "beta_warmup": 0,
    "batch_size":  16,
    "lr":          1e-4,
    "categories":  ["bottle"],
}

# Clases borderline: mapa activa pero localización imprecisa.
# Re-entrenar con LATENT_DIM=64 para forzar más compresión
# y concentrar el error en zonas anómalas.
# NO incluye: screw, grid, toothbrush, zipper, carpet (fallo estructural).
RERUN_CATEGORIES = ["bottle", "capsule", "leather", "pill", "transistor", "wood"]

RERUN = {
    "img_size":    256,
    "epochs":      50,
    "latent_dim":  64,
    "beta":        0.5,
    "lambda1":     0.5,
    "lambda2":     0.5,
    "lambda3":     0.0,
    "beta_warmup": 0,
    "batch_size":  16,
    "lr":          1e-4,
    "categories":  RERUN_CATEGORIES,
}

FULL = {
    "img_size":    256,
    "epochs":      50,
    "latent_dim":  128,
    "beta":        0.5,
    "lambda1":     0.5,
    "lambda2":     0.5,
    "lambda3":     0.0,
    "beta_warmup": 0,
    "batch_size":  16,
    "lr":          1e-4,
    "categories": [
        "bottle", "cable", "capsule", "carpet", "grid",
        "hazelnut", "leather", "metal_nut", "pill", "screw",
        "tile", "toothbrush", "transistor", "wood", "zipper",
    ],
}

# Perceptual loss + KL annealing.
# Calibracion de lambda3:
#   F.mse_loss en espacio VGG para imagenes aleatorias: ~15-20.
#   F.mse_loss pixel-level (lambda1*MSE + lambda2*SSIM): ~0.1-0.3.
#   Para que el termino perceptual aporte ~20-30% del loss pixel:
#     lambda3 = 0.01 → contribucion ~0.15-0.20. Escala comparable.
#     lambda3 = 0.1  → contribucion ~1.5-2.0. Domina — demasiado.
#
# beta_warmup=10: primeros 10 epochs el decoder aprende sin presion KL.
#   Evita posterior collapse temprano y mejora calidad de reconstruccion inicial.
#
# batch_size=8: VGG hace forward sobre la reconstruccion (256x256).
#   relu1_2 genera [8, 64, 256, 256] = 134MB solo de activaciones.
#   Con batch=16 puede OOM en 8GB VRAM.
FULL_V2 = {
    "img_size":    256,
    "epochs":      50,
    "latent_dim":  128,
    "beta":        0.5,
    "lambda1":     0.5,
    "lambda2":     0.5,
    "lambda3":     0.01,  # peso del loss perceptual VGG (calibrado vs escala VGG)
    "beta_warmup": 10,    # KL annealing: beta 0->0.5 en los primeros 10 epochs
    "batch_size":  8,     # reducido por VRAM (VGG forward en 256x256)
    "lr":          1e-4,
    "categories": [
        "bottle", "cable", "capsule", "carpet", "grid",
        "hazelnut", "leather", "metal_nut", "pill", "screw",
        "tile", "toothbrush", "transistor", "wood", "zipper",
    ],
}

DATA_ROOT             = "Data"
CKPT_DIR              = "checkpoints"
RESULTS_DIR           = "results"
CKPT_DIR_RERUN        = "checkpoints_rerun"
RESULTS_DIR_RERUN     = "results_rerun"
RESULTS_DIR_IMPROVED  = "results_improved"
RESULTS_DIR_RERUN_IMP = "results_rerun_improved"
CKPT_DIR_V2           = "checkpoints_v2"         # perceptual loss + KL annealing
RESULTS_DIR_V2        = "results_v2"


# ─────────────────────────────────────────────
# Pipeline por clase
# ─────────────────────────────────────────────

def run_category(cfg: dict, category: str, device: torch.device,
                 rerun: bool = False, eval_only: bool = False,
                 v2: bool = False, perceptual_loss=None):
    print(f"\n{'='*50}")
    print(f"  Categoría: {category}")
    print(f"{'='*50}")

    # Datos
    train_loader, test_loader = get_dataloaders(
        DATA_ROOT, category,
        img_size=cfg["img_size"],
        batch_size=cfg["batch_size"],
    )

    # Modelo
    model = ConvVAE(latent_dim=cfg["latent_dim"], img_size=cfg["img_size"]).to(device)
    print(f"  Parámetros: {sum(p.numel() for p in model.parameters()):,}")

    if v2:
        ckpt_base = CKPT_DIR_V2
    elif rerun:
        ckpt_base = CKPT_DIR_RERUN
    else:
        ckpt_base = CKPT_DIR
    ckpt_path = Path(ckpt_base) / f"{category}.pth"

    if eval_only:
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint no encontrado: {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
        print(f"  Checkpoint cargado: {ckpt_path}")
    else:
        print(f"  Entrenando {cfg['epochs']} épocas...")
        train(
            model, train_loader, device,
            epochs=cfg["epochs"],
            lr=cfg["lr"],
            beta=cfg["beta"],
            lambda1=cfg["lambda1"],
            lambda2=cfg["lambda2"],
            lambda3=cfg.get("lambda3", 0.0),
            beta_warmup=cfg.get("beta_warmup", 0),
            perceptual_loss=perceptual_loss,
        )
        ckpt_path.parent.mkdir(exist_ok=True)
        torch.save(model.state_dict(), ckpt_path)

    print(f"  Evaluando...")
    results = evaluate_category(
        model, test_loader, device, category,
        lambda1=cfg["lambda1"],
        lambda2=cfg["lambda2"],
    )

    if v2:
        results_base = RESULTS_DIR_V2
    elif eval_only:
        results_base = RESULTS_DIR_RERUN_IMP if rerun else RESULTS_DIR_IMPROVED
    else:
        results_base = RESULTS_DIR_RERUN if rerun else RESULTS_DIR

    save_figures(
        model, test_loader, device, category,
        best_threshold=results["best_threshold"],
        out_dir=results_base,
        global_min=results["global_min"],
        global_max=results["global_max"],
        pixel_min=results["pixel_min"],
        pixel_max=results["pixel_max"],
        feat_min=results["feat_min"],
        feat_max=results["feat_max"],
        feat_alpha=results["feat_alpha"],
        lambda1=cfg["lambda1"],
        lambda2=cfg["lambda2"],
    )
    save_f1_curve(results, results_base)

    return results


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full",      action="store_true", help="Entrenamiento real (todas las clases)")
    parser.add_argument("--rerun",     action="store_true", help="Re-entrena clases borderline con LATENT_DIM=64")
    parser.add_argument("--eval-only", action="store_true", help="Re-evalua checkpoints existentes (sin entrenar)")
    parser.add_argument("--v2",        action="store_true", help="Entrena con perceptual loss + KL annealing")
    parser.add_argument("--category",  type=str, default=None, help="Solo esta clase")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.v2:
        cfg = FULL_V2
    elif args.rerun:
        cfg = RERUN
    elif args.full:
        cfg = FULL
    else:
        cfg = SMOKE

    if args.category:
        categories = [args.category]
    else:
        categories = cfg["categories"]

    eval_only = args.eval_only

    if eval_only:
        modo = "EVAL-ONLY (multi-scale SSIM)"
    elif args.v2:
        modo = "V2 (perceptual loss + KL annealing)"
    elif args.full:
        modo = "FULL"
    else:
        modo = "SMOKE TEST"

    print(f"\nModo: {modo}")
    print(f"IMG_SIZE={cfg['img_size']} | EPOCHS={cfg['epochs']} | LATENT_DIM={cfg['latent_dim']}")
    if args.v2:
        print(f"lambda3={cfg['lambda3']} | beta_warmup={cfg['beta_warmup']} | batch={cfg['batch_size']}")
    print(f"Clases: {categories}")

    # Instanciar PerceptualLoss una sola vez (carga VGG16 una vez)
    perc_loss = None
    if args.v2 and cfg.get("lambda3", 0.0) > 0.0:
        print("  Cargando VGG16 para perceptual loss...")
        perc_loss = PerceptualLoss().to(device)
        print("  VGG16 listo.")

    all_results = {}

    for category in categories:
        try:
            results = run_category(cfg, category, device,
                                   rerun=args.rerun, eval_only=eval_only,
                                   v2=args.v2, perceptual_loss=perc_loss)
            all_results[category] = {
                "best_f1":        results["best_f1"],
                "best_threshold": results["best_threshold"],
            }
        except Exception as e:
            print(f"  Error en {category}: {e}")
            all_results[category] = {"best_f1": None, "error": str(e)}

    # ─── Tabla resumen ───
    print(f"\n{'='*50}")
    print(f"  RESULTADOS FINALES")
    print(f"{'='*50}")
    print(f"  {'Categoría':<15} {'F1':>8} {'Umbral':>10}")
    print(f"  {'-'*35}")

    f1s = []
    for cat, r in all_results.items():
        f1  = r.get("best_f1")
        thr = r.get("best_threshold")
        if f1 is not None:
            print(f"  {cat:<15} {f1:>8.4f} {thr:>10.3f}")
            f1s.append(f1)
        else:
            print(f"  {cat:<15} {'ERROR':>8}")

    if f1s:
        print(f"  {'-'*35}")
        print(f"  {'PROMEDIO':<15} {sum(f1s)/len(f1s):>8.4f}")

    # Guardar JSON
    if args.v2:
        results_base = RESULTS_DIR_V2
    elif eval_only:
        results_base = RESULTS_DIR_RERUN_IMP if args.rerun else RESULTS_DIR_IMPROVED
    else:
        results_base = RESULTS_DIR_RERUN if args.rerun else RESULTS_DIR
    out_json = Path(results_base) / "results.json"
    out_json.parent.mkdir(exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Resultados guardados en {out_json}")


if __name__ == "__main__":
    main()
