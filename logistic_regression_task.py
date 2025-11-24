import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

from lab1.utils_lab1 import (
    collect_image_paths,
    plot_sample_images,
    class_distribution,
    split_dataset,
    deduplicate_train,
    paths_to_arrays,
)


def run_lab1(data_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    paths_labels = collect_image_paths(data_dir)
    print(f"Collected {len(paths_labels)} images")

    plot_sample_images(paths_labels, os.path.join(output_dir, 'sample_images.png'))
    class_distribution(paths_labels, os.path.join(output_dir, 'class_distribution.png'))

    train, val, test = split_dataset(paths_labels)
    print(f"Split sizes: train={len(train)}, val={len(val)}, test={len(test)}")

    train_clean, removed = deduplicate_train(train, val, test)
    print(f"Removed {removed} duplicates. New train size: {len(train_clean)}")

    X_val, y_val_labels = paths_to_arrays(val)
    X_test, y_test_labels = paths_to_arrays(test)

    labels_sorted = sorted(list({lbl for _, lbl in train + val + test}))
    label_to_idx = {l: i for i, l in enumerate(labels_sorted)}

    y_val = np.array([label_to_idx[l] for l in y_val_labels])
    y_test = np.array([label_to_idx[l] for l in y_test_labels])

    train_sizes = [50, 100, 1000, 50000]
    results = {}

    for ts in train_sizes:
        use_n = min(ts, len(train_clean))
        print(f"Training with {use_n} samples")
        X_train, y_train_labels = paths_to_arrays(train_clean, max_items=use_n)
        y_train = np.array([label_to_idx[l] for l in y_train_labels])

        if X_train.shape[0] == 0:
            results[ts] = None
            continue

        solver = 'saga' if X_train.shape[0] > 2000 else 'lbfgs'
        clf = LogisticRegression(
            multi_class='multinomial',
            solver=solver,
            max_iter=1000
        )
        clf.fit(X_train, y_train)

        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        results[ts] = acc

        print(f"Accuracy (train={use_n}) → {acc:.4f}")
        joblib.dump(clf, os.path.join(output_dir, f"lr_{use_n}.joblib"))

    with open(os.path.join(output_dir, 'lab1_report.txt'), 'w', encoding='utf-8') as f:
        f.write('Lab1 Results\n')
        f.write(f"Train size after dedup: {len(train_clean)}\n")
        for ts, acc in results.items():
            f.write(f"{ts}: {acc}\n")

    print("Lab1 finished")
