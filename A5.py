import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

CONFIG = {
    "dataset": "cifar10",
    "data_dir": "./my_dataset",

    "img_height": 32,
    "img_width":  32,
    "channels":   3,      

    "batch_size":    32,
    "epochs":        20,
    "learning_rate": 1e-3,
    "val_split":     0.2,   

    "dropout_rate": 0.5,

    "model_save_path": "cnn_model.keras",
    "plot_save_path":  "training_history.png",
}

def load_data(cfg):
    """Load and preprocess dataset. Returns (x_train, y_train, x_test, y_test, class_names)."""
    dataset = cfg["dataset"].lower()

    if dataset == "cifar10":
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
        class_names = ["airplane","automobile","bird","cat","deer",
                       "dog","frog","horse","ship","truck"]
        x_train, x_test = x_train / 255.0, x_test / 255.0

    elif dataset == "mnist":
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        class_names = [str(i) for i in range(10)]
        x_train = x_train[..., np.newaxis] / 255.0
        x_test  = x_test[...,  np.newaxis] / 255.0

    elif dataset == "fashion_mnist":
        (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
        class_names = ["T-shirt","Trouser","Pullover","Dress","Coat",
                       "Sandal","Shirt","Sneaker","Bag","Ankle boot"]
        x_train = x_train[..., np.newaxis] / 255.0
        x_test  = x_test[...,  np.newaxis] / 255.0

    elif dataset == "custom":
        return load_custom_data(cfg)

    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    y_train = y_train.flatten()
    y_test  = y_test.flatten()
    print(f"Train: {x_train.shape}  |  Test: {x_test.shape}")
    print(f"Classes ({len(class_names)}): {class_names}")
    return x_train, y_train, x_test, y_test, class_names


def load_custom_data(cfg):
    """Load images from a directory tree (one subfolder per class)."""
    from tensorflow.keras.utils import image_dataset_from_directory

    img_size = (cfg["img_height"], cfg["img_width"])
    color_mode = "grayscale" if cfg["channels"] == 1 else "rgb"

    train_ds = image_dataset_from_directory(
        cfg["data_dir"],
        validation_split=cfg["val_split"],
        subset="training",
        seed=42,
        image_size=img_size,
        batch_size=cfg["batch_size"],
        color_mode=color_mode,
    )
    val_ds = image_dataset_from_directory(
        cfg["data_dir"],
        validation_split=cfg["val_split"],
        subset="validation",
        seed=42,
        image_size=img_size,
        batch_size=cfg["batch_size"],
        color_mode=color_mode,
    )

    class_names = train_ds.class_names
    normalization = layers.Rescaling(1.0 / 255)
    train_ds = train_ds.map(lambda x, y: (normalization(x), y))
    val_ds   = val_ds.map(lambda x, y:   (normalization(x), y))
    return train_ds, val_ds, class_names


def build_augmentation():
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomTranslation(0.1, 0.1),
    ], name="augmentation")

def build_model(cfg, num_classes):
    """Builds a CNN with 3 conv blocks + dense head."""
    h, w, c = cfg["img_height"], cfg["img_width"], cfg["channels"]
    inputs = keras.Input(shape=(h, w, c))

    x = build_augmentation()(inputs)

    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(cfg["dropout_rate"])(x)

    activation = "sigmoid" if num_classes == 2 else "softmax"
    outputs = layers.Dense(num_classes, activation=activation)(x)

    model = keras.Model(inputs, outputs, name="CNN_Classifier")
    return model


def train_model(model, cfg, x_train, y_train, x_test, y_test):
    loss = "binary_crossentropy" if model.output_shape[-1] == 1 else "sparse_categorical_crossentropy"

    model.compile(
        optimizer=keras.optimizers.Adam(cfg["learning_rate"]),
        loss=loss,
        metrics=["accuracy"],
    )
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3, verbose=1),
        keras.callbacks.ModelCheckpoint(cfg["model_save_path"], save_best_only=True, verbose=1),
    ]

    history = model.fit(
        x_train, y_train,
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        validation_split=cfg["val_split"],
        callbacks=callbacks,
    )
    return history


def evaluate_model(model, x_test, y_test, class_names, cfg):
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\n✅ Test Accuracy: {acc:.4f}  |  Test Loss: {loss:.4f}")

    y_pred = np.argmax(model.predict(x_test), axis=1)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.show()
    print("Confusion matrix saved → confusion_matrix.png")


def plot_history(history, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history.history["accuracy"],     label="Train")
    ax1.plot(history.history["val_accuracy"], label="Validation")
    ax1.set_title("Accuracy"); ax1.set_xlabel("Epoch")
    ax1.legend(); ax1.grid(True)

    ax2.plot(history.history["loss"],     label="Train")
    ax2.plot(history.history["val_loss"], label="Validation")
    ax2.set_title("Loss"); ax2.set_xlabel("Epoch")
    ax2.legend(); ax2.grid(True)

    plt.suptitle("Training History", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Training history saved → {save_path}")


def visualize_predictions(model, x_test, y_test, class_names, n=10):
    indices = np.random.choice(len(x_test), n, replace=False)
    images  = x_test[indices]
    labels  = y_test[indices]
    preds   = np.argmax(model.predict(images), axis=1)

    cols = 5
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i >= n:
            ax.axis("off"); continue
        img = images[i].squeeze()
        ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
        color = "green" if preds[i] == labels[i] else "red"
        ax.set_title(f"True: {class_names[labels[i]]}\nPred: {class_names[preds[i]]}", color=color, fontsize=9)
        ax.axis("off")

    plt.suptitle("Sample Predictions  (green=correct, red=wrong)", fontsize=12)
    plt.tight_layout()
    plt.savefig("sample_predictions.png", dpi=150)
    plt.show()
    print("Sample predictions saved → sample_predictions.png")


def main():
    tf.random.set_seed(42)
    np.random.seed(42)

    cfg = CONFIG
    result = load_data(cfg)

    if cfg["dataset"] == "custom":
        train_ds, val_ds, class_names = result
        num_classes = len(class_names)
        model = build_model(cfg, num_classes)
        loss = "sparse_categorical_crossentropy"
        model.compile(
            optimizer=keras.optimizers.Adam(cfg["learning_rate"]),
            loss=loss, metrics=["accuracy"]
        )
        callbacks = [
            keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3),
            keras.callbacks.ModelCheckpoint(cfg["model_save_path"], save_best_only=True),
        ]
        history = model.fit(train_ds, validation_data=val_ds,
                            epochs=cfg["epochs"], callbacks=callbacks)
        plot_history(history, cfg["plot_save_path"])
        return

    x_train, y_train, x_test, y_test, class_names = result
    num_classes = len(class_names)
    model = build_model(cfg, num_classes)

    history = train_model(model, cfg, x_train, y_train, x_test, y_test)

    plot_history(history, cfg["plot_save_path"])
    evaluate_model(model, x_test, y_test, class_names, cfg)

    visualize_predictions(model, x_test, y_test, class_names)

    print(f"\n🎉 Done! Model saved → {cfg['model_save_path']}")


if __name__ == "__main__":
    main()