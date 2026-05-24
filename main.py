"""
main.py — Orquesta el pipeline completo.

Uso:
  python main.py                         # smoke test (bottle, 5 epochs, 128px)
  python main.py --full                  # entrenamiento real (todas las clases)
  python main.py --full --category wood  # solo una clase en full
"""

import argparse
import json
import os
import torch
from pathlib import Path

from src.dataset  import get_dataloaders
from src.model    import ConvVAE
from src.train    import train
from src.evaluate import evaluate_category, save_figures, save_f1_curve

# ─────────────────────────────────────────────
# Configuración
# ─────────────────────────────────────────────

SMOKE = {
    "img_size":   128,
    "epochs":     5,
    "latent_dim": 128,
    "beta":       0.5,
    "lambda1":    0.5,
    "lambda2":    0.5,
    "batch_size": 16,
    "lr":         1e-4,
    "categories": ["bottle"],
}

FULL = {
    "img_size":   256,
    "epochs":     50,
    "latent_dim": 128,
    "beta":       0.5,
    "lambda1":    0.5,
    "lambda2":    0.5,
    "batch_size": 16,
    "lr":         1e-4,
    "categories": [
        "bottle", "cable", "capsule", "carpet", "grid",
        "hazelnut", "leather", "metal_nut", "pill", "screw",
        "tile", "toothbrush", "transistor", "wood", "zipper",
    ],
}

DATA_ROOT   = "Data"
CKPT_DIR    = "checkpoints"
RESULTS_DIR = "results"


# ─────────────────────────────────────────────
# Pipeline por clase
# ─────────────────────────────────────────────

def run_category(cfg: dict, category: str, device: torch.device):
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

    # Entrenamiento
    print(f"  Entrenando {cfg['epochs']} épocas...")
    history = train(
        model, train_loader, device,
        epochs=cfg["epochs"],
        lr=cfg["lr"],
        beta=cfg["beta"],
        lambda1=cfg["lambda1"],
        lambda2=cfg["lambda2"],
    )

    # Guardar checkpoint
    ckpt_path = Path(CKPT_DIR) / f"{category}.pth"
    ckpt_path.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), ckpt_path)

    # Evaluación
    print(f"  Evaluando...")
    results = evaluate_category(
        model, test_loader, device, category,
        lambda1=cfg["lambda1"],
        lambda2=cfg["lambda2"],
    )

    # Figuras
    save_figures(
        model, test_loader, device, category,
        best_threshold=results["best_threshold"],
        out_dir=RESULTS_DIR,
        lambda1=cfg["lambda1"],
        lambda2=cfg["lambda2"],
    )
    save_f1_curve(results, RESULTS_DIR)

    return results


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full",     action="store_true", help="Entrenamiento real (todas las clases)")
    parser.add_argument("--category", type=str, default=None, help="Solo esta clase (con --full)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cfg = FULL if args.full else SMOKE

    if args.category:
        categories = [args.category]
    else:
        categories = cfg["categories"]

    print(f"\nModo: {'FULL' if args.full else 'SMOKE TEST'}")
    print(f"IMG_SIZE={cfg['img_size']} | EPOCHS={cfg['epochs']} | LATENT_DIM={cfg['latent_dim']}")
    print(f"Clases: {categories}")

    all_results = {}

    for category in categories:
        try:
            results = run_category(cfg, category, device)
            all_results[category] = {
                "best_f1":        results["best_f1"],
                "best_threshold": results["best_threshold"],
            }
        except Exception as e:
            print(f"  ❌ Error en {category}: {e}")
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
    out_json = Path(RESULTS_DIR) / "results.json"
    out_json.parent.mkdir(exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Resultados guardados en {out_json}")


if __name__ == "__main__":
    main()
