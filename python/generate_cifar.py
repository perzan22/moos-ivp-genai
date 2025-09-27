import os
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

LATENT_DIM = 100
N_CLASS = 10
TAGS = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
        'Dog', 'Frog', 'Horse', 'Ship', 'Truck']


def load_generator(path):
    return tf.keras.models.load_model(path, compile=False)


def generate_images(generator, label, num_samples=1, seed=None, outdir="generated_cifar"):
    os.makedirs(outdir, exist_ok=True)

    rng = np.random.default_rng(seed)
    noise = rng.normal(size=(num_samples, LATENT_DIM)).astype(np.float32)

    labels = np.ones((num_samples, 1)) * label

    preds = generator.predict([noise, labels], verbose=0)
    preds = (preds + 1) / 2.0  # [-1,1] -> [0,1]

    for i in range(num_samples):
        img = preds[i]
        plt.imshow(img)
        plt.axis("off")
        plt.title(TAGS[label])
        save_path = os.path.join(outdir, f"class{label}_{i+1}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"[OK] Saved: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Ścieżka do pliku generatora (.keras)")
    parser.add_argument("--label", type=int, required=True, choices=list(range(10)),
                        help="Którą klasę chcesz wygenerować (0-9)")
    parser.add_argument("--n", type=int, default=5, help="Ile obrazków wygenerować")
    parser.add_argument("--seed", type=int, default=None, help="Seed generatora losowego")
    parser.add_argument("--outdir", type=str, default="generated_cifar", help="Folder na wygenerowane obrazki")
    args = parser.parse_args()

    print(f"[INFO] Ładuję model z {args.model}")
    generator = load_generator(args.model)

    print(f"[INFO] Generuję klasę {args.label} ({TAGS[args.label]})")
    generate_images(generator, label=args.label, num_samples=args.n, seed=args.seed, outdir=args.outdir)
