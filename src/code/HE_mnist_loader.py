import csv
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def load_mnist(version='small', size_tr=2000, size_te=500, plot=False):


    """
    Had to delete the full dataset due to git lfs size limits.
    """
    if version == 'full':
        raise ValueError("Full dataset no longer available. Please use version='small' instead.")

    elif version == 'small':
        with open('data/mnist_train_small.csv') as csvfile:
            readCSV = csv.reader(csvfile, delimiter=',')

            x_tr = np.zeros((size_tr, 784))
            y_tr = np.zeros((size_tr))

            for i, row in enumerate(tqdm(readCSV, total=size_tr, desc='Loading small training data')):
                x_tr[i] = np.array(row[1:], dtype=np.int)
                y_tr[i] = np.array(row[0], dtype=np.int)

        with open("data/mnist_test_small.csv") as csvfile:
            readCSV = csv.reader(csvfile, delimiter=',')

            x_te = np.zeros((size_te, 784))
            y_te = np.zeros((size_te))

            for i, row in enumerate(tqdm(readCSV, total=size_te, desc='Loading small test data')):
                x_te[i] = np.array(row[1:], dtype=np.int)
                y_te[i] = np.array(row[0], dtype=np.int)

        x_tr = x_tr.reshape(size_tr, 1, 28, 28)
        x_te = x_te.reshape(size_te, 1, 28, 28)

    if plot:
        idx = np.random.randint(0, size_tr-1, 9)
        plt.figure(1)
        for i in range(9):
            plt.subplot(3, 3, 1+i)
            plt.imshow(x_tr[idx[i], 0])
            plt.axis('off')

        plt.tight_layout()
        plt.show()

    return x_tr, y_tr, x_te, y_te

def plot_mnist_results(data, pred, label):
    idx = np.random.randint(0, label.shape[0], 9)
    plt.figure(1)
    for i in range(9):
        plt.subplot(3, 3, 1+i)
        plt.title('Pred:'+ str(pred[idx[i]]) + ' Label:' + str(int(label[idx[i]])))
        plt.imshow(data[idx[i], 0])
        plt.axis('off')

    plt.tight_layout()
    plt.show()
