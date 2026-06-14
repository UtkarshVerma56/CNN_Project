# utils/visualization.py

import matplotlib.pyplot as plt


def plot_loss(history):

    plt.figure(figsize=(10,5))

    plt.plot(
        history["train_loss"],
        label="Train"
    )

    plt.plot(
        history["val_loss"],
        label="Validation"
    )

    plt.legend()
    plt.show()
