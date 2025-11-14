import jax
import struct
import pickle 

import numpy as np
import pandas as pd
import scipy as sp
import jax.numpy as jnp

from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler


def standardize_data(
    train_data,
    val_data,
    test_data 
):
    """
    Standardize_data data.

    params:
    - train_data (jnp.array): training data.
    - val_data (jnp.array): validation data.
    - test_data (jnp.array): test data.
    
    returns: 
    - train_data (jnp.array): standardized training data.
    - val_data (jnp.array): standardized validation data.
    - test_data (jnp.array): standardized test data.
    """
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    val_data = scaler.transform(val_data)
    test_data = scaler.transform(test_data)

    return train_data, val_data, test_data


def standardize_image_data(
    train_img,
    val_img,
    test_img
):  
    """
    Standardize image data.

    params:
    - train_img (jnp.array): training images.
    - val_img (jnp.array): validation images.
    - test_img (jnp.array): test images.

    returns:
    - train_img (jnp.array): standardized training images.
    - val_img (jnp.array): standardized validation images.
    - test_img (jnp.array): standardized test images.
    """
    mean = np.mean(train_img, axis=(0,), keepdims=True)
    std = np.std(train_img, axis=(0,), keepdims=True) + 1e-10
    _train_img = (train_img - mean) / std
    _val_img = (val_img - mean) / std
    _test_img = (test_img - mean) / std

    return _train_img, _val_img, _test_img


def read_uci_data(
    dataset_name
):
    """
    Read UCI data.

    params:
    - dataset_name (string): name of the dataset.

    return:
    - X (jnp.array): feature matrix.
    - y (jnp.array): target matrix.
    """
    file_name = "../Data/" + dataset_name + ".csv"
    df = pd.read_csv(file_name, sep=',', header=None)
    arr = np.random.permutation(df.to_numpy())
    X, y = arr[:,:-1], arr[:,-1].reshape(-1, 1)

    return X, y


def read_toy_data(
    config
):
    """
    Read toy data.

    params:
    - config (dict): configuration dictionary.

    returns:
    - X (jnp.array): feature matrix.
    - y (jnp.array): target matrix.
    """
    key = jax.random.PRNGKey(0)

    # Load configuration
    feature_dim = config["data"]["feature_dim"]
    n_samples = config["data"]["n_samples"]

    # Load data
    if config["data"]["name"] == "truncated_sine":
        X, y = _truncated_sine(key, n_samples, feature_dim)
    elif config["data"]["name"] == "two_moons":
        X, y = _two_moons(n_samples)
    else:
        raise Exception("Unknown toy dataset.")
    
    return X, y


def read_image_data(
    dataset_name
):
    """
    Read image data.
    """
    # Load data
    if dataset_name == "mnist":
        X_train, X_test, y_train, y_test = _mnist()
    elif dataset_name == "fashion_mnist":
        X_train, X_test, y_train, y_test = _fashion_mnist()
    elif dataset_name == "cifar10":
        X_train, X_test, y_train, y_test = _cifar10()
    elif dataset_name == "svhn":
        X_train, X_test, y_train, y_test = _svhn()
    elif dataset_name == "kmnist":
        X_train, X_test, y_train, y_test = _kmnist()
    elif dataset_name == "cifar100":
        X_train, X_test, y_train, y_test = _cifar100()
    else:
        raise Exception("Unknown image dataset.")
    
    return X_train, X_test, y_train, y_test


def _mnist():
    """
    Read MNIST data.

    returns:
    - X_train (jnp.array): training feature matrix.
    - X_test (jnp.array): test feature matrix.
    - y_train (jnp.array): training target matrix.
    - y_test (jnp.array): test target matrix.
    """
    # Load train images
    with open("../Data/mnist/train_images",'rb') as file:
        magic, size, nrows, ncols = struct.unpack('>4i', file.read(16))
        X_train = np.fromfile(file, dtype=np.dtype(np.uint8).newbyteorder('>'))
        X_train = X_train.reshape((size, nrows, ncols, 1))
        X_train = jnp.float64(X_train) / 255

    # Load test images
    with open("../Data/mnist/test_images",'rb') as file:
        magic, size, nrows, ncols = struct.unpack('>4i', file.read(16))
        X_test = np.fromfile(file, dtype=np.dtype(np.uint8).newbyteorder('>'))
        X_test = X_test.reshape((size, nrows, ncols, 1))
        X_test = jnp.float64(X_test) / 255

    # Load train labels
    with open("../Data/mnist/train_labels", 'rb') as file:
        magic, size  = struct.unpack('>2i', file.read(8))
        y_train = np.fromfile(file, dtype=np.dtype(np.uint8)).newbyteorder(">")    
        y_train = y_train.reshape((size, -1))

    # Load test labels
    with open("../Data/mnist/test_labels", 'rb') as file:
        magic, size  = struct.unpack('>2i', file.read(8))
        y_test = np.fromfile(file, dtype=np.dtype(np.uint8)).newbyteorder(">")    
        y_test = y_test.reshape((size, -1))

    return X_train, X_test, y_train, y_test


def _kmnist():
    """
    Read KMNIST data.
    https://github.com/rois-codh/kmnist

    returns:
    - X_train (jnp.array): training feature matrix.
    - X_test (jnp.array): test feature matrix.
    - y_train (jnp.array): training target matrix.
    - y_test (jnp.array): test target matrix.
    """
    IMG_ROWS, IMG_COLS = 28, 28
    
    # Load train images
    X_train = np.load('../Data/kmnist/kmnist-train-imgs.npz')['arr_0']
    X_train = X_train.reshape((X_train.shape[0], IMG_ROWS, IMG_COLS, 1))
    X_train = jnp.float64(X_train) / 255

    # Load test images
    X_test = np.load('../Data/kmnist/kmnist-test-imgs.npz')['arr_0']
    X_test = X_test.reshape((X_test.shape[0], IMG_ROWS, IMG_COLS, 1))
    X_test = jnp.float64(X_test) / 255

    # Load train labels
    y_train = np.load('../Data/kmnist/kmnist-train-labels.npz')['arr_0']
    y_train = y_train.reshape((y_train.shape[0], 1))

    # Load test labels
    y_test = np.load('../Data/kmnist/kmnist-test-labels.npz')['arr_0']
    y_test = y_test.reshape((y_test.shape[0], 1))

    return X_train, X_test, y_train, y_test


def _fashion_mnist():
    """
    Read Fashion MNIST data.

    returns:
    - X_train (jnp.array): training feature matrix.
    - X_test (jnp.array): test feature matrix.
    - y_train (jnp.array): training target matrix.
    - y_test (jnp.array): test target matrix.
    """
    # Load train images
    with open("../Data/fashion_mnist/train_images",'rb') as file:
        magic, size, nrows, ncols = struct.unpack('>4i', file.read(16))
        if magic != 2051:
            raise RuntimeError(f"../Data/mnist/train_images not an MNIST image set.")
        X_train = np.fromfile(file, dtype=np.dtype(np.uint8).newbyteorder('>'))
        X_train = X_train.reshape((size, nrows, ncols, 1))
        X_train = jnp.float64(X_train) / 255

    # Load test images
    with open("../Data/fashion_mnist/test_images",'rb') as file:
        magic, size, nrows, ncols = struct.unpack('>4i', file.read(16))
        X_test = np.fromfile(file, dtype=np.dtype(np.uint8).newbyteorder('>'))
        X_test = X_test.reshape((size, nrows, ncols, 1))
        X_test = jnp.float64(X_test) / 255

    # Load train labels
    with open("../Data/fashion_mnist/train_labels", 'rb') as file:
        magic, size  = struct.unpack('>2i', file.read(8))
        y_train = np.fromfile(file, dtype=np.dtype(np.uint8)).newbyteorder(">")    
        y_train = y_train.reshape((size, -1))

    # Load test labels
    with open("../Data/fashion_mnist/test_labels", 'rb') as file:
        magic, size  = struct.unpack('>2i', file.read(8))
        y_test = np.fromfile(file, dtype=np.dtype(np.uint8)).newbyteorder(">")    
        y_test = y_test.reshape((size, -1))

    return X_train, X_test, y_train, y_test


def _cifar10():
    """
    Read CIFAR 10 data.

    returns:
    - X_train (jnp.array): training feature matrix.
    - X_test (jnp.array): test feature matrix.
    - y_train (jnp.array): training target matrix.
    - y_test (jnp.array): test target matrix.
    """
    # Load train images
    train_images, train_labels = [], []
    for i in range(1,6):
        f_name = f"../Data/cifar10/data_batch_{i}"
        with open(f_name, 'rb') as file:
            data_dict = pickle.load(file, encoding='latin1')
            train_images += [data_dict['data'].reshape(-1, 3, 32, 32).transpose(0,2,3,1)] # (10000, 32, 32, 3)
            train_labels += [data_dict['labels']] # (10000,1)
    X_train = np.concatenate(train_images, axis=0)
    y_train = np.concatenate(train_labels, axis=0).reshape(-1, 1)

    # Load test images
    with open("../Data/cifar10/test_batch", 'rb') as file:
        data_dict = pickle.load(file, encoding='latin1')
        X_test = data_dict['data'].reshape(-1, 3, 32, 32).transpose(0,2,3,1) # (10000, 32, 32, 3)
        y_test = np.array(data_dict['labels']).reshape(-1, 1) # (10000,1)

    # Preprocess data
    X_train = jnp.float64(X_train) / 255
    X_test = jnp.float64(X_test) / 255
    
    return X_train, X_test, y_train, y_test


def _cifar100():
    """
    Read CIFAR 100 data.

    returns:
    - X_train (jnp.array): training feature matrix.
    - X_test (jnp.array): test feature matrix.
    - y_train (jnp.array): training target matrix.
    - y_test (jnp.array): test target matrix.
    """
    # Load train data
    with open("../Data/cifar100/train", 'rb') as file:
        dict = pickle.load(file, encoding='latin1')
    X_train = dict['data'].reshape(-1, 3, 32, 32).transpose(0,2,3,1) # (50000, 32, 32, 3)
    y_train = np.array(dict['fine_labels']).reshape(-1, 1) # (50000,1)

    # Load test data
    with open("../Data/cifar100/test", 'rb') as file:
        dict = pickle.load(file, encoding='latin1')
    X_test = dict['data'].reshape(-1, 3, 32, 32).transpose(0,2,3,1) # (10000, 32, 32, 3)
    y_test = np.array(dict['fine_labels']).reshape(-1, 1) # (10000,1)

    # Preprocess data
    X_train = jnp.float64(X_train) / 255
    X_test = jnp.float64(X_test) / 255

    return X_train, X_test, y_train, y_test


def _svhn():
    """
    Read SVHN data.

    returns:
    - X_train (jnp.array): training feature matrix.
    - X_test (jnp.array): test feature matrix.
    - y_train (jnp.array): training target matrix.
    - y_test (jnp.array): test target matrix.
    """
    # Load train images
    train_svhn = sp.io.loadmat("../Data/svhn/train_32x32.mat")
    X_train = np.transpose(train_svhn['X'], [3, 0, 1, 2])
    y_train = train_svhn['y'].reshape(-1, 1)
    y_train[np.where(y_train==10)] = 0

    # Load test images
    test_svhn = sp.io.loadmat("../Data/svhn/test_32x32.mat")
    X_test = np.transpose(test_svhn['X'], [3, 0, 1, 2])
    y_test = test_svhn['y'].reshape(-1, 1)
    y_test[np.where(y_test==10)] = 0

    # Preprocess data
    X_train = jnp.float64(X_train) / 255
    X_test = jnp.float64(X_test) / 255

    return X_train, X_test, y_train, y_test


def _truncated_sine(
    key,
    n_samples,
    feature_dim
):
    """
    Generated data from truncated sine function.

    params:
    - key (jax.random.PRNGKey): random key.
    - n_samples (int): number of samples.
    - feature_dim (int): feature dimension.

    returns:
    - X (jnp.array): feature matrix.
    - y (jnp.array): target matrix.
    """
    key1, key2, key3 = jax.random.split(key, num=3)

    # Features
    X1 = jax.random.uniform(key1, minval=-1, maxval=-0.5, shape=(n_samples//2, feature_dim))
    X2 = jax.random.uniform(key2, minval=0.5, maxval=1, shape=(n_samples//2, feature_dim))
    X = jnp.concatenate([X1, X2], axis=0)

    # Targets
    eps = 0.1*jax.random.normal(key3, shape=(n_samples,))
    y = jnp.sin(2*np.pi*X.mean(axis=-1)) + eps

    # Format
    X = X.reshape(-1, feature_dim)
    y = y.reshape(-1, 1)

    return X, y


def _two_moons(
    n_samples
):
    """
    Generate two moons dataset.
    
    params:
    - n_samples (int): number of samples.

    returns:
    - X (jnp.array): feature matrix.
    - y (jnp.array): target matrix.
    """
    X, y = make_moons(n_samples, shuffle=False, noise=0.1)
    X = StandardScaler().fit_transform(X)
    y = y.reshape(-1, 1)

    return X, y

