import os
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

from lab2.utils_lab2 import prepare_flattened_dataset


def build_mlp(
        input_shape,
        hidden_layers=[512, 256, 128],
        activation='relu',
        dropout=0.5,
        l2_reg=1e-4
):
    model = keras.Sequential()
    model.add(layers.Input(shape=(input_shape,)))

    for units in hidden_layers:
        model.add(
            layers.Dense(
                units,
                activation=activation,
                kernel_regularizer=keras.regularizers.l2(l2_reg)
            )
        )
        if dropout > 0:
            model.add(layers.Dropout(dropout))

    model.add(layers.Dense(10, activation='softmax'))
    return model


def run_lab2(data_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    (X_train, y_train), (X_val, y_val), (X_test, y_test), _ = prepare_flattened_dataset(data_dir)
    print("Shapes:", X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape)

    mean = np.mean(X_train, axis=0)

    X_train = (X_train - mean) / 128.0
    X_val = (X_val - mean) / 128.0
    X_test = (X_test - mean) / 128.0

    model = build_mlp(
        input_shape=X_train.shape[1],
        hidden_layers=[512, 256, 128],
        activation='relu',
        dropout=0.5,
        l2_reg=1e-4
    )

    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3,
        decay_steps=10000,
        decay_rate=0.9
    )

    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=256
    )

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {acc:.4f}")

    model.save(os.path.join(output_dir, "mlp_model"))

    import json
    with open(os.path.join(output_dir, "lab2_report.json"), "w") as f:
        json.dump(
            {
                "test_accuracy": float(acc),
                "history": {
                    k: [float(x) for x in v]
                    for k, v in history.history.items()
                }
            }, f
        )

    print("Lab2 finished.")
