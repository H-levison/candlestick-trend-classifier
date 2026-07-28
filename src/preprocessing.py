"""Data loading, augmentation, and feature-interpretation visualizations
for the Up/Down candlestick classifier."""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy import ndimage
import tensorflow as tf

IMG_SIZE = (224, 224)
BATCH_SIZE = 32


def load_datasets(train_dir, test_dir, img_size=IMG_SIZE, batch_size=BATCH_SIZE, val_split=0.2, seed=42):
    """Loads train/val/test datasets from the Up/Down folder structure.

    Images are returned as raw float32 pixels in [0, 255]. Scaling is applied
    inside the model itself via `tf.keras.applications.mobilenet_v2.preprocess_input`
    (see model.py), which maps to [-1, 1] as MobileNetV2's ImageNet weights
    expect -- a plain 1/255 rescale would hurt transfer-learning performance.
    Keeping this in the model (not here) guarantees training, evaluation, and
    single-image inference all apply identical preprocessing.
    """
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=val_split,
        subset="training",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="binary",
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=val_split,
        subset="validation",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="binary",
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="binary",
        shuffle=False,
    )

    class_names = train_ds.class_names  # ["Down", "Up"]

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(autotune)
    val_ds = val_ds.cache().prefetch(autotune)
    test_ds = test_ds.cache().prefetch(autotune)

    return train_ds, val_ds, test_ds, class_names


def get_augmentation_layer():
    """Light augmentation appropriate for chart images: no flips (would
    reverse time direction / candle meaning), just slight zoom/rotation/
    brightness jitter. No-ops automatically at inference time."""
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomRotation(0.02),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomBrightness(0.1),
        ],
        name="augmentation",
    )


def preprocess_image(image_path_or_bytes, img_size=IMG_SIZE):
    """Loads a single image (path or raw bytes) and resizes it to a
    (1, H, W, 3) float32 array in [0, 255], ready for model.predict().
    Scaling to [-1, 1] happens inside the model (see load_datasets docstring).
    """
    if isinstance(image_path_or_bytes, (bytes, bytearray)):
        import io

        img = Image.open(io.BytesIO(image_path_or_bytes))
    else:
        img = Image.open(image_path_or_bytes)

    img = img.convert("RGB").resize(img_size)
    arr = np.asarray(img, dtype=np.float32)
    return np.expand_dims(arr, axis=0)


# ---------------------------------------------------------------------------
# Visual-trend labeling
# ---------------------------------------------------------------------------

ALPHA_THRESHOLD = 10  # a pixel counts as "ink" if its alpha exceeds this


def detect_visual_trend(image_path):
    """Labels a candlestick chart image by the visual trend it shows: does
    the staircase of candle bodies rise or fall, left to right, across the
    10 candles actually drawn in the image.

    This is classical image processing (a least-squares fit of ink-pixel
    row-position vs column-index), not a trained model -- deliberately, so
    it can be used as a label source without circularity. Image row 0 is
    the top, so a negative slope (row decreases left-to-right, i.e. price
    rises on screen) means "Up"; a positive slope means "Down".

    These source PNGs are RGBA with a fully-transparent background
    (alpha=0). Converting straight to RGB does not composite onto white --
    it keeps whatever RGB values happen to be stored under the transparent
    pixels (here, (0,0,0), i.e. black), which would make the entire
    background register as "ink" under a brightness threshold. Using the
    alpha channel directly avoids that.

    Returns (label, slope).
    """
    img = Image.open(image_path)
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    arr = np.asarray(img)

    is_ink = arr[:, :, 3] > ALPHA_THRESHOLD  # (H, W) bool, from alpha channel

    cols_with_ink = np.where(is_ink.any(axis=0))[0]
    if len(cols_with_ink) < 2:
        return "Up", 0.0  # degenerate/blank image fallback

    row_positions = [np.where(is_ink[:, x])[0].mean() for x in cols_with_ink]

    slope, _intercept = np.polyfit(cols_with_ink, row_positions, 1)
    label = "Down" if slope > 0 else "Up"
    return label, float(slope)


# ---------------------------------------------------------------------------
# Feature interpretation / visualization functions
# ---------------------------------------------------------------------------


def plot_color_channel_dominance(image_paths_by_class, sample_size=150):
    """Feature 1: average R/G/B channel intensity per class.

    Story: candlestick chart renderers typically color bullish (up) candles
    green and bearish (down) candles red. If that convention holds in this
    dataset, the Up class should show a visibly higher mean green channel
    and the Down class a higher mean red channel.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    means = {}
    for cls, paths in image_paths_by_class.items():
        sample = paths[:sample_size]
        channel_sums = np.zeros(3)
        for p in sample:
            arr = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32)
            channel_sums += arr.mean(axis=(0, 1))
        means[cls] = channel_sums / max(len(sample), 1)

    classes = list(means.keys())
    r_vals = [means[c][0] for c in classes]
    g_vals = [means[c][1] for c in classes]
    b_vals = [means[c][2] for c in classes]

    x = np.arange(len(classes))
    width = 0.25
    ax.bar(x - width, r_vals, width, label="Red", color="#d62728")
    ax.bar(x, g_vals, width, label="Green", color="#2ca02c")
    ax.bar(x + width, b_vals, width, label="Blue", color="#1f77b4")
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylabel("Mean channel intensity (0-255)")
    ax.set_title("Color Channel Dominance: Bullish (Up) vs Bearish (Down)")
    ax.legend()
    fig.tight_layout()
    return fig, means


def plot_edge_detection(image_path):
    """Feature 2: Sobel edge map isolating candle wicks/bodies from the
    flat chart background.

    Story: wicks are thin, high-gradient vertical lines and bodies are
    filled rectangles with strong horizontal edges at top/bottom — the
    Sobel map makes that structure explicit instead of implicit in raw
    pixels, which is what the CNN is effectively learning to detect.
    """
    img = Image.open(image_path).convert("L").resize(IMG_SIZE)
    arr = np.asarray(img, dtype=np.float32)

    sobel_x = ndimage.sobel(arr, axis=0)
    sobel_y = ndimage.sobel(arr, axis=1)
    edges = np.hypot(sobel_x, sobel_y)
    edges = (edges / edges.max() * 255) if edges.max() > 0 else edges

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(arr, cmap="gray")
    axes[0].set_title("Original (grayscale)")
    axes[0].axis("off")
    axes[1].imshow(edges, cmap="gray")
    axes[1].set_title("Sobel Edge Map (wicks/bodies)")
    axes[1].axis("off")
    fig.tight_layout()
    return fig, edges


def plot_class_distribution_and_intensity(image_paths_by_class, sample_size=150):
    """Feature 3: class balance + pixel intensity distribution per class.

    Story: confirms whether the dataset is balanced enough to train on
    without class-weighting, and whether Up/Down images differ in overall
    brightness/contrast (which would hint at trivially separable shortcuts
    the model might learn instead of genuine pattern structure).
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    classes = list(image_paths_by_class.keys())
    counts = [len(image_paths_by_class[c]) for c in classes]
    axes[0].bar(classes, counts, color=["#d62728", "#2ca02c"])
    axes[0].set_title("Class Distribution")
    axes[0].set_ylabel("Number of images")
    for i, c in enumerate(counts):
        axes[0].text(i, c, str(c), ha="center", va="bottom")

    for cls, paths in image_paths_by_class.items():
        sample = paths[:sample_size]
        intensities = []
        for p in sample:
            arr = np.asarray(Image.open(p).convert("L"), dtype=np.float32)
            intensities.append(arr.mean())
        axes[1].hist(intensities, bins=20, alpha=0.6, label=cls)
    axes[1].set_title("Mean Pixel Intensity Distribution")
    axes[1].set_xlabel("Mean grayscale intensity")
    axes[1].set_ylabel("Image count")
    axes[1].legend()

    fig.tight_layout()
    return fig, dict(zip(classes, counts))
