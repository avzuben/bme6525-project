import numpy as np
import matplotlib.pyplot as plt

ifig = 0

history = []
history_fine = []
accuracy = []

epochs = np.arange(1, 101)

for i in range(5):
    history.append(np.load('./cnn-history/cnn-history-' + str(i) + '.npy', allow_pickle=True).item())
    history_fine.append(np.load('./cnn-history/cnn-history--fine-' + str(i) + '.npy', allow_pickle=True).item())

    accuracy.append(np.max(history[i]['val_accuracy'] + history_fine[i]['val_accuracy']))

    ifig += 1
    plt.figure(ifig)
    plt.title('loss vs epochs - fold #' + str(i))
    plt.plot(epochs, history[i]['loss'] + history_fine[i]['loss'], label='training')
    plt.plot(epochs, history[i]['val_loss'] + history_fine[i]['val_loss'], label='validation')
    plt.axvline(x=50.5, color='black', linestyle='dashed')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.grid(which='both', axis='both')
    plt.legend()
    plt.show()

    ifig += 1
    plt.figure(ifig)
    plt.title('accuracy vs epochs - fold #' + str(i))
    plt.plot(epochs, history[i]['accuracy'] + history_fine[i]['accuracy'], label='training')
    plt.plot(epochs, history[i]['val_accuracy'] + history_fine[i]['val_accuracy'], label='validation')
    plt.axvline(x=50.5, color='black', linestyle='dashed')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.grid(which='both', axis='both')
    plt.legend()
    plt.show()

