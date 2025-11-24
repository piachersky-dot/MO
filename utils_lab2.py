import numpy as np

from lab1.utils_lab1 import collect_image_paths, image_to_array


def prepare_flattened_dataset(
        data_dir,
        train_n=200000,
        val_n=10000,
        test_n=19000,
        seed=42
):
    paths_labels = collect_image_paths(data_dir)

    import random
    random.seed(seed)
    random.shuffle(paths_labels)

    test = paths_labels[:min(test_n, len(paths_labels))]
    val = paths_labels[min(test_n, len(paths_labels)):min(test_n + val_n, len(paths_labels))]
    train = paths_labels[min(test_n + val_n, len(paths_labels)):min(test_n + val_n + train_n, len(paths_labels))]

    def load(pl):
        X, y = [], []
        for p, lbl in pl:
            arr = image_to_array(p)
            if arr is None:
                continue
            X.append(arr.flatten())
            y.append(lbl)
        return np.array(X, dtype=np.float32), np.array(y)

    X_train, y_train_lbl = load(train)
    X_val, y_val_lbl = load(val)
    X_test, y_test_lbl = load(test)

    labels_sorted = sorted(list({lbl for _, lbl in paths_labels}))
    label_to_idx = {l: i for i, l in enumerate(labels_sorted)}

    y_train = np.array([label_to_idx[l] for l in y_train_lbl])
    y_val = np.array([label_to_idx[l] for l in y_val_lbl])
    y_test = np.array([label_to_idx[l] for l in y_test_lbl])

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), label_to_idx
