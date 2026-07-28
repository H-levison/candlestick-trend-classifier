"""CNN (MobileNetV2 transfer learning) for Up/Down candlestick classification:
build, train, evaluate, save/load, and fine-tune (retrain)."""

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss

from src.preprocessing import IMG_SIZE, get_augmentation_layer, load_datasets


def build_model(img_size=IMG_SIZE, dropout_rate=0.5, learning_rate=1e-3, unfreeze_base=False):
    """MobileNetV2 backbone (ImageNet weights, frozen by default) + a small
    classification head. Dropout for regularization, Adam with a configurable
    learning rate (lowered for fine-tuning during retraining)."""
    base = tf.keras.applications.MobileNetV2(
        input_shape=img_size + (3,), include_top=False, weights="imagenet"
    )
    base.trainable = unfreeze_base

    inputs = tf.keras.Input(shape=img_size + (3,))
    x = get_augmentation_layer()(inputs)
    # Equivalent to mobilenet_v2.preprocess_input's default "tf" mode
    # (x/127.5 - 1 -> [-1, 1]), as a native Rescaling layer instead of a
    # Lambda wrapping an imported function -- Lambda holds a raw reference
    # to the function's module, which isn't deep-copyable/picklable and
    # breaks partway through training in this TF/Keras version.
    x = tf.keras.layers.Rescaling(
        scale=1.0 / 127.5, offset=-1.0, name="mobilenetv2_preprocess"
    )(x)
    x = base(x, training=unfreeze_base)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs, name="candlestick_classifier")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def get_early_stopping(patience=3):
    return tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=patience, restore_best_weights=True
    )


def train_model(model, train_ds, val_ds, epochs=15, patience=3):
    return model.fit(
        train_ds, validation_data=val_ds, epochs=epochs, callbacks=[get_early_stopping(patience)]
    )


def evaluate_model(model, test_ds):
    """Runs full evaluation on a held-out dataset. Accuracy/Precision/Recall
    come from Keras metrics during training, but are recomputed here via
    sklearn (alongside F1, which isn't a portable built-in Keras metric
    across TF versions) directly from predictions, so this function is the
    single source of truth for reported metrics.

    Returns (metrics_dict, y_true, y_pred, y_prob) -- the last three are for
    building the confusion matrix / ROC curve in the notebook.
    """
    y_true = np.concatenate([y.numpy() for _, y in test_ds], axis=0).ravel().astype(int)
    y_prob = model.predict(test_ds).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
    }
    return metrics, y_true, y_pred, y_prob


def save_model(model, path):
    model.save(path)


def load_model(path):
    return tf.keras.models.load_model(path)


def retrain(model_path, train_dir, test_dir, epochs=5, learning_rate=1e-5, patience=2):
    """Loads the existing model, fine-tunes it on the current data/train/
    directory (which may include newly uploaded samples) with a lower
    learning rate and early stopping, evaluates before/after on test_dir,
    and overwrites the .h5 file. Returns before/after metrics so callers
    (API response, Streamlit summary) can show the retraining had an effect
    without needing a live training log stream.
    """
    model = load_model(model_path)
    train_ds, val_ds, test_ds, _ = load_datasets(train_dir, test_dir)

    before_metrics, *_ = evaluate_model(model, test_ds)

    model.optimizer.learning_rate.assign(learning_rate)
    train_model(model, train_ds, val_ds, epochs=epochs, patience=patience)

    after_metrics, *_ = evaluate_model(model, test_ds)
    save_model(model, model_path)

    return {"before": before_metrics, "after": after_metrics}
