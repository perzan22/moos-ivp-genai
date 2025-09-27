import os
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

IMG_SIZE = 128
LATENT_DIM = 100
CONDITION_DIM = 8   # zmień na tyle ile masz w swoim modelu (np. 8 dla start_x, start_y i waypointów)

def load_generator(path):
    return tf.keras.models.load_model(path, compile=False)

def generate_images(generator, conditions, num_samples=1, seed=None, outdir="generated_moos"):
    os.makedirs(outdir, exist_ok=True)

    rng = np.random.default_rng(seed)
    noise = rng.normal(size=(num_samples, LATENT_DIM)).astype(np.float32)

    cond = np.array(conditions, dtype=np.float32)
    if cond.ndim == 1:
        cond = np.tile(cond, (num_samples, 1))  # powiel jeśli tylko jeden warunek

    preds = generator([noise, cond], training=False)
    preds = (preds + 1) / 2.0  # [-1,1] -> [0,1]

    for i in range(num_samples):
        img = preds[i].numpy().squeeze()
        plt.imshow(img, cmap="gray" if img.ndim == 2 else None)
        plt.axis("off")
        save_path = os.path.join(outdir, f"sample_{i+1}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"[OK] Saved: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Ścieżka do pliku generatora (.keras)")
    parser.add_argument("--conds", type=float, nargs="+", required=True,
                        help="Warunki wejściowe (lista liczb float, np. start_x start_y wp1_x wp1_y wp2_x wp2_y ...)")
    parser.add_argument("--n", type=int, default=1, help="Ile obrazków wygenerować")
    parser.add_argument("--seed", type=int, default=None, help="Seed generatora losowego")
    parser.add_argument("--outdir", type=str, default="generated", help="Folder na wygenerowane obrazki")
    args = parser.parse_args()

    print(f"[INFO] Ładuję model z {args.model}")
    generator = load_generator(args.model)

    print(f"[INFO] Warunki wejściowe: {args.conds}")
    generate_images(generator, conditions=args.conds, num_samples=args.n, seed=args.seed, outdir=args.outdir)
