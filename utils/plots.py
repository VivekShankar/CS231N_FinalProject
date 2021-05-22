import numpy as np
import matplotlib.pyplot as plt

def plot_loss_and_accuracy_curves(model_name, loss, acc, loss_v, acc_v):
    plt.figure()
    epochs = np.arange(len(loss))+1
    plt.plot(epochs, loss, label='{} - training'.format(model_name))
    plt.plot(epochs, loss_v, label='{} - validation'.format(model_name))
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Pretrained {} Loss".format(model_name))
    plt.legend()
    plt.show()
    plt.figure()
    plt.plot(epochs, acc, label='{} - training'.format(model_name))
    plt.plot(epochs, acc_v, label='{} - validation'.format(model_name))
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Pretrained {} Accuracy".format(model_name))
    plt.legend()
    plt.show()