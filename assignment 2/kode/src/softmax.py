import numpy as np
import matplotlib as mpl

mpl.use('TkAgg')
import matplotlib.pyplot as plt
from kode.src import mnist
import tqdm
import copy


def should_early_stop(validation_loss, num_steps=3):
    """
    Returns true if the validation loss increases
    or stays the same for num_steps.
    --
    validation_loss: List of floats
    num_steps: integer
    """
    if len(validation_loss) < num_steps + 1:
        return False

    is_increasing = [validation_loss[i] <= validation_loss[i + 1] for i in range(-num_steps - 1, -1)]
    return sum(is_increasing) == len(is_increasing)


def train_val_split(X, Y, val_percentage):
    """
    Selects samples from the dataset randomly to be in the validation set.
    Also, shuffles the train set.
    --
    X: [N, num_features] numpy vector,
    Y: [N, 1] numpy vector
    val_percentage: amount of data to put in validation set
    """
    dataset_size = X.shape[0]
    idx = np.arange(0, dataset_size)
    np.random.shuffle(idx)

    train_size = int(dataset_size * (1 - val_percentage))
    idx_train = idx[:train_size]
    idx_val = idx[train_size:]
    X_train, Y_train = X[idx_train], Y[idx_train]
    X_val, Y_val = X[idx_val], Y[idx_val]
    return X_train, Y_train, X_val, Y_val


def shuffle_training_data(X_train, Y_train):
    """
    method shuffles the training data
    Stolen from https://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison
    :return:
    """
    assert len(X_train) == len(Y_train)
    p = np.random.permutation(len(X_train))
    X_train = X_train[p]
    Y_train = Y_train[p]
    return X_train, Y_train


def onehot_encode(Y, n_classes=10):
    onehot = np.zeros((Y.shape[0], n_classes))
    onehot[np.arange(0, Y.shape[0]), Y] = 1
    return onehot


def bias_trick(X):
    """
    X: shape[batch_size, num_features(784)]
    -- 
    Returns [batch_size, num_features+1 ]
    """
    return np.concatenate((X, np.ones((len(X), 1))), axis=1)


def check_gradient(X, targets, w, epsilon, computed_gradient, index):
    """
    Computes the numerical approximation for the gradient of w,
    w.r.t. the input X and target vector targets.
    Asserts that the computed_gradient from backpropagation is
    correct w.r.t. the numerical approximation.
    --
    X: shape: [batch_size, num_features(784+1)]. Input batch of images
    targets: shape: [batch_size, num_classes]. Targets/label of images
    w: shape: [num_classes, num_features]. Weight from input->output
    epsilon: Epsilon for numerical approximation (See assignment)
    computed_gradient: Gradient computed from backpropagation. Same shape as w.
    """
    print("Checking gradient...")
    dw = np.zeros_like(w[index])
    for k in range(w[index].shape[0]):
        for j in range(w[index].shape[1]):
            new_weight1, new_weight2 = copy.deepcopy(w), copy.deepcopy(w)
            new_weight1[index][k, j] += epsilon
            new_weight2[index][k, j] -= epsilon
            loss1 = cross_entropy_loss(X, targets, new_weight1)
            loss2 = cross_entropy_loss(X, targets, new_weight2)
            dw[k, j] = (loss1 - loss2) / (2 * epsilon)
    maximum_absolute_difference = abs(computed_gradient - dw).max()
    assert maximum_absolute_difference <= epsilon ** 2, "Absolute error was: {}".format(maximum_absolute_difference)


def softmax(a):
    """
    Applies the softmax activation function for the vector a.
    --
    a: shape: [batch_size, num_classes]. Activation of the output layer before activation
    --
    Returns: [batch_size, num_classes] numpy vector
    """
    a_exp = np.exp(a)
    return a_exp / a_exp.sum(axis=1, keepdims=True)


def sigmoid(a):
    """
    Applies the sigmoid function for vector a
    :param a:
    :return:
    """
    return 1 / (1 + np.exp(-a))


def imp_sigmoid(a):
    return 1.7159 * np.tanh(2 / 3 * a)


def forward(X, w):
    """
    Performs a forward pass through the network
    --
    X: shape: [batch_size, num_features(784+1)] numpy vector. Input batch of images
    w: shape: [num_classes, num_features] numpy vector. Weight from input->output
    --
    Returns: [batch_size, num_classes] numpy vector
    """
    a = X
    for i in range(0, len(w) - 1):
        zj = a.dot(w[i].T)  # calculates zj = X*w1^T
        if improved_sigmoid:
            a = imp_sigmoid(zj)
        else:
            a = sigmoid(zj)  # calculates f(zj)

    a = a.dot(w[1].T)  # calculate zk = f(zj)*W2^t
    return softmax(a)


def calculate_accuracy(X, targets, w):
    """
    Calculated the accuracy of the network.
    ---
    X: shape: [batch_size, num_features(784+1)] numpy vector. Input batch of images
    targets: shape: [batch_size, num_classes] numpy vector. Targets/label of images
    w: shape: [num_classes, num_features] numpy vector. Weight from input->output
    --
    Returns float
    """
    output = forward(X, w)
    predictions = output.argmax(axis=1)
    targets = targets.argmax(axis=1)
    return (predictions == targets).mean()


def cross_entropy_loss(X, targets, w):
    """
    Computes the cross entropy loss given the input vector X and the target vector.
    ---
    X: shape: [batch_size, num_features(784+1)] numpy vector. Input batch of images
    targets: shape: [batch_size, num_classes] numpy vector. Targets/label of images
    w: shape: [num_classes, num_features] numpy vector. Weight from input->output
    --
    Returns float
    """
    output = forward(X, w)
    assert output.shape == targets.shape, "det gikk galt"
    log_y = np.log(output)
    cross_entropy = -targets * log_y
    return cross_entropy.mean()


def gradient_descent(X, targets, w, learning_rate, should_check_gradient, old_dw):
    """
    Performs gradient descents for all weights in the network.
    ---
    X: shape: [batch_size, num_features(784+1)] numpy vector. Input batch of images
    targets: shape: [batch_size, num_classes] numpy vector. Targets/label of images
    w: list, shape: [w1 .. wn] where w1 .. contains [ .... TODO: fill out
    --
    Returns updated w, with same shape
    """

    # Since we are taking the .mean() of our loss, we get the normalization factor to be 1/(N*C)
    # If you take loss.sum(), the normalization factor is 1
    # The normalization factor is identical for all weights in the network (For multi-layer neural-networks as well.)
    normalization_factor_w2 = X.shape[0] * targets.shape[1]  # batch_size * num_classes
    normalization_factor_w1 = X.shape[0] * units

    normalization_factors = []
    for i in range(0, layers):
        normalization_factors.append(X.shape[0] * units)
    normalization_factors.append(X.shape[0] * targets.shape[1])

    dw = []
    if improved_sigmoid:
        aj = imp_sigmoid(X.dot(w[0].T))
    else:
        aj = sigmoid(X.dot(w[0].T))
    for i in range(0, layers - 1):
        # calculate aj for the last hidden layer
        aj = X
        if improved_sigmoid:
            aj = imp_sigmoid(X.dot(w[i].T))
        else:
            aj = sigmoid(X.dot(w[i].T))

    outputs = forward(X, w)  # shape: (batch_size, outputs)
    delta_k = - (targets - outputs)
    dw2 = delta_k.T.dot(aj)
    dw2 = dw2 / normalization_factor_w2  # Normalize gradient equally as we do with the loss

    # handle gradient descent for internal nodes
    ones = np.ones((batch_size, units))  # not really sure about the size of this
    if improved_sigmoid:  # implements the derivative of the improved sigmoid (logistic function)
        f_derived = 2.28786 / np.cosh(4 / 3 * X.dot(w[0].T) + ones) # f'()
    else:
        f_derived = np.multiply(aj, ones - aj)  # piecewise multiplication of aj and (ones - aj)

    sum_wkj_delta_k = delta_k.dot(w[1])
    dw1 = np.multiply(f_derived, sum_wkj_delta_k)
    dw1 = np.dot(dw1.T, X)  # deltaj * Xi
    dw1 = dw1 / normalization_factor_w1

    # assert dw.shape == w.shape, "dw shape was: {}. Expected: {}".format(dw.shape, w.shape)

    if should_check_gradient:
        check_gradient(X, targets, w, 1e-2, dw2, 1)

    if momentum:
        dw1 = learning_rate * dw1 - mu * old_dw[0]  # subtracts since change in W is the negative of the old dw
        dw2 = learning_rate * dw2 - mu * old_dw[1]
    else:
        dw1 = learning_rate * dw1
        dw2 = learning_rate * dw2

    w[0] = w[0] - dw1
    w[1] = w[1] - dw2
    # w = w - learning_rate * dw
    return w, [dw1, dw2]


# load data
X_train, Y_train, X_test, Y_test = mnist.load()

# Pre-process data
X_train, X_test = X_train / 127.5 - 1, X_test / 127.5 - 1
X_train = bias_trick(X_train)
X_test = bias_trick(X_test)
Y_train, Y_test = onehot_encode(Y_train), onehot_encode(Y_test)

X_train, Y_train, X_val, Y_val = train_val_split(X_train, Y_train, 0.1)

# Hyperparameters
batch_size = 32
learning_rate = 0.25
num_batches = X_train.shape[0] // batch_size
should_gradient_check = False
check_step = num_batches // 10
max_epochs = 15 # use 15 while testing
layers = 1  # number of hidden layers
units = 32  # number of units per hidden layer
shuffle_before_epoch = True
improved_sigmoid = True
improved_initialization = True
momentum = True
mu = 0.9
early_stop_steps = 5

# Tracking variables
TRAIN_LOSS = []
TEST_LOSS = []
VAL_LOSS = []
TRAIN_ACC = []
TEST_ACC = []
VAL_ACC = []


def train_loop():
    """
    trains the necessary weight matrixes
    :return:
    """
    w = []
    if improved_initialization:
        w.append(np.random.normal(0, 1 / np.sqrt(X_train.shape[1]), (units, X_train.shape[1])))
        for i in range(0, layers - 1):
            w.append(np.random.normal(0, 1 / np.sqrt(units), (units, units)))
            # w1 = np.random.normal(0, 1 / np.sqrt(X_train.shape[1]), (units, X_train.shape[1]))
        # w3 = np.random.normal(0, 1 / np.sqrt(units), (Y_train.shape[1], units))
        w.append(np.random.normal(0, 1 / np.sqrt(units), (Y_train.shape[1], units)))
    else:
        w.append(np.random.uniform(-1, 1, (units, X_train.shape[1])))
        w.append(np.random.uniform(-1, 1, (Y_train.shape[1], units)))
    dw = []
    for list in w:
        dw.append(np.zeros(list.shape))

    for e in range(max_epochs):  # Epochs
        if shuffle_before_epoch:
            X_train_shuffled, Y_train_shuffled = shuffle_training_data(X_train, Y_train)
        else:
            X_train_shuffled, Y_train_shuffled = X_train, Y_train
        for i in tqdm.trange(num_batches):
            # gets the data that will be used in this specific batch of training
            X_batch = X_train_shuffled[i * batch_size:(i + 1) * batch_size]
            Y_batch = Y_train_shuffled[i * batch_size:(i + 1) * batch_size]

            w, dw = gradient_descent(X_batch,
                                     Y_batch,
                                     w,
                                     learning_rate,
                                     should_gradient_check,
                                     dw)
            # handles saving of data
            if i % check_step == 0:
                # Loss
                TRAIN_LOSS.append(cross_entropy_loss(X_train, Y_train, w))
                TEST_LOSS.append(cross_entropy_loss(X_test, Y_test, w))
                VAL_LOSS.append(cross_entropy_loss(X_val, Y_val, w))

                TRAIN_ACC.append(calculate_accuracy(X_train, Y_train, w))
                VAL_ACC.append(calculate_accuracy(X_val, Y_val, w))
                TEST_ACC.append(calculate_accuracy(X_test, Y_test, w))
                if should_early_stop(VAL_LOSS, early_stop_steps):
                    print(VAL_LOSS[-4:])
                    print("early stopping.")
                    return w
    return w


w = train_loop()

plt.plot(TRAIN_LOSS, label="Training loss")
plt.plot(TEST_LOSS, label="Testing loss")
plt.plot(VAL_LOSS, label="Validation loss")
plt.legend()
plt.ylim([0, 1.5])
plt.show()

plt.clf()
plt.plot(TRAIN_ACC, label="Training accuracy")
plt.plot(TEST_ACC, label="Testing accuracy")
plt.plot(VAL_ACC, label="Validation accuracy")
plt.ylim([0.7, 1.0])
plt.legend()
plt.show()

# # old
# plt.clf()
# w = w[:, :-1]  # Removes bias
# w = w.reshape(10, 28, 28)
# w = np.concatenate(w, axis=0)
# plt.imshow(w, cmap="gray")
# plt.show()
