import os
from pathlib import Path
from PIL import Image
import numpy as np
import hashlib
import matplotlib.pyplot as plt


def image_to_array(path, size=(28, 28)):
    try:
        img = Image.open(path).convert("L")
        if img.size != size:
            img = img.resize(size, Image.ANTIALIAS)
        arr = np.array(img, dtype=np.float32) / 255.0
        return arr
    except Exception:
        return None


def md5_of_image(path):
    try:
        with open(path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception:
        return None


def collect_image_paths(root_dir):
    root = Path(root_dir)
    paths = []
    for label_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        label = label_dir.name
        for f in label_dir.iterdir():
            if f.is_file():
                paths.append((str(f), label))
    return paths


def plot_sample_images(paths_labels, out_path):
    samples = paths_labels[:10]
    imgs = [image_to_array(p) for p, _ in samples]
    labels = [lbl for _, lbl in samples]
    n = len(imgs)

    plt.figure(figsize=(n * 1.4, 2))
    for i, (img, lbl) in enumerate(zip(imgs, labels)):
        plt.subplot(1, n, i + 1)
        plt.imshow(img, cmap="gray")
        plt.title(lbl)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def class_distribution(paths_labels, out_path):
    from collections import Counter
    ctr = Counter([lbl for _, lbl in paths_labels])
    labels = sorted(ctr.keys())
    counts = [ctr[l] for l in labels]
    plt.figure(figsize=(8, 4))
    plt.bar(labels, counts)
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.title("Class distribution")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def split_dataset(paths_labels, train_n=200000, val_n=10000, test_n=19000, seed=42):
    import random
    random.seed(seed)
    items = paths_labels.copy()
    random.shuffle(items)

    total = len(items)

    test = items[:min(test_n, total)]
    val = items[min(test_n, total):min(test_n + val_n, total)]
    train = items[min(test_n + val_n, total):min(test_n + val_n + train_n, total)]

    return train, val, test


def deduplicate_train(train, val, test):
    md5_map = {}
    for p, _ in (val + test):
        h = md5_of_image(p)
        if h:
            md5_map.setdefault(h, []).append(p)

    new_train = []
    removed = 0
    for p, lbl in train:
        h = md5_of_image(p)
        if h and h in md5_map:
            removed += 1
            continue
        new_train.append((p, lbl))

    return new_train, removed


def paths_to_arrays(paths_labels, max_items=None):
    X, y = [], []
    for i, (p, lbl) in enumerate(paths_labels):
        if max_items and i >= max_items:
            break
        arr = image_to_array(p)
        if arr is None:
            continue
        X.append(arr.flatten())
        y.append(lbl)
    return np.array(X, dtype=np.float32), np.array(y)
