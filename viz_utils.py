import matplotlib.pyplot as plt


def plot_losses(history, path_save=None, show_val=True):
    plt.plot(history["loss_train"], label="train")
    if show_val:
        plt.plot(history["loss_val"], label="val")
    plt.grid()
    plt.title("Loss vs. epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    if path_save is not None:
        plt.savefig(path_save)
    plt.show()


def plot_costs(history, path_save=None, show_val=True):
    plt.plot(history["cost_train"], label="train")
    if show_val:
        plt.plot(history["cost_val"], label="val")
    plt.grid()
    plt.title("Cost vs. epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Cost")
    plt.legend()
    if path_save is not None:
        plt.savefig(path_save)
    plt.show()


def plot_accuracies(history, path_save=None, show_val=True):
    plt.plot(history["accuracy_train"], label="train")
    if show_val:
        plt.plot(history["accuracy_val"], label="val")
    plt.grid()
    plt.title("Accuracy vs. epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    if path_save is not None:
        plt.savefig(path_save)
    plt.show()


def plot_lrs(history, path_save=None):
    plt.plot(history["lr"], label="lr")
    plt.grid()
    plt.title("Learning rate vs. epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Learning rate")
    plt.legend()
    if path_save is not None:
        plt.savefig(path_save)
    plt.show()