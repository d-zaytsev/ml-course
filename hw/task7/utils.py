import numpy as np


def softmax(predictions):
    """
    Computes probabilities from scores.

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions -
        probability for every class, 0..1
    """
    m = np.max(predictions, axis=1)
    m = m[:, np.newaxis]
    exps = np.exp(predictions - m)
    div = np.sum(exps, axis=1)
    div = div[:, np.newaxis]
    return exps / div


def cross_entropy_loss(probabilities, target_index):
    """
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    """
    return -np.log(
        probabilities[
            np.arange(probabilities.shape[0]),
            target_index.reshape(-1, target_index.shape[0]),
        ]
    ).sum()


def softmax_with_cross_entropy(predictions, target_index):
    """
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    """
    probs = softmax(predictions)
    loss = cross_entropy_loss(probs, target_index)
    dprediction = probs
    dprediction[
        np.arange(dprediction.shape[0]), target_index.reshape(-1, target_index.shape[0])
    ] -= 1
    return loss, dprediction


def l2_regularization(W, reg_strength):
    """
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    """

    # Накладывает штраф на большие веса
    # "reg" - насколько сильно штраф будет влиять на вес

    loss = reg_strength * np.sum(np.square(W[:-1]))
    grad = 2 * reg_strength * W
    grad[-1] = 0

    return loss, grad


def check_gradient(f, x, delta=1e-5, tol=1e-4):
    """
    Checks the implementation of analytical gradient by comparing
    it to numerical gradient using two-point formula

    Arguments:
      f: function that receives x and computes value and gradient
      x: np array, initial point where gradient is checked
      delta: step to compute numerical gradient
      tol: tolerance for comparing numerical and analytical gradient

    Return:
      bool indicating whether gradients match or not
    """

    assert isinstance(x, np.ndarray)
    assert x.dtype == float

    orig_x = x.copy()
    fx, analytic_grad = f(x)
    assert np.all(np.isclose(orig_x, x, tol)), (
        "Functions shouldn't modify input variables"
    )

    assert analytic_grad.shape == x.shape
    analytic_grad = analytic_grad.copy()

    # We will go through every dimension of x and compute numeric
    # derivative for it
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        ix = it.multi_index
        analytic_grad_at_ix = analytic_grad[ix]
        h = np.zeros(x.shape)
        h[ix] = delta
        fxh, _ = f(x + h)
        numeric_grad_at_ix = (fxh - fx) / delta

        if not np.isclose(numeric_grad_at_ix, analytic_grad_at_ix, tol):
            print(
                "Gradients are different at %s. Analytic: %2.5f, Numeric: %2.5f"
                % (ix, analytic_grad_at_ix, numeric_grad_at_ix)
            )
            return False

        it.iternext()

    return True


def ReLU(X):
    return np.maximum(0, X)


def ReLU_backward(d_output, X):
    dX = d_output.copy()
    dX[X <= 0] = 0

    return dX


class LinearSoftmaxClassifier:
    def __init__(self, input_num, classes_num, hidden_neurons_num):
        self.__W1 = 0.001 * np.random.randn(input_num, hidden_neurons_num)
        self.__W2 = 0.001 * np.random.randn(hidden_neurons_num, classes_num)

        self.__b1 = np.zeros(hidden_neurons_num)
        self.__b2 = np.zeros(classes_num)

    def fit(self, X, y, batch_size=100, learning_rate=1e-7, reg=1e-5, epochs=1):
        """
        Trains linear classifier

        Arguments:
          X, np array (num_samples, num_features) - training data
          y, np array of int (num_samples) - labels
          batch_size, int - batch size to use
          learning_rate, float - learning rate for gradient descent
          reg, float - L2 regularization strength
          epochs, int - number of epochs
        """

        num_train = X.shape[0]

        for epoch in range(epochs):
            shuffled_indices = np.arange(num_train)
            np.random.shuffle(shuffled_indices)

            sections = np.arange(batch_size, num_train, batch_size)
            batches_indices = np.array_split(shuffled_indices, sections)

            for batch_idx in batches_indices:
                batchX, batchY = X[batch_idx], y[batch_idx]

                # Forward pass
                Z1 = ReLU(batchX @ self.__W1 + self.__b1)
                Z2 = Z1 @ self.__W2 + self.__b2

                loss, dZ2 = softmax_with_cross_entropy(Z2, batchY)

                l2_loss1, l2_grad1 = l2_regularization(self.__W1, reg)
                l2_loss2, l2_grad2 = l2_regularization(self.__W2, reg)

                loss += l2_loss1 + l2_loss2
                print(f"Epoch {epoch}, loss: {loss}")

                # Backward pass

                dZ1 = dZ2 @ self.__W2.T
                dZ1 = ReLU_backward(dZ1, Z1)

                dW2 = (Z1.T @ dZ2) + l2_grad2
                dW1 = (batchX.T @ dZ1) + l2_grad1

                db2 = np.sum(dZ2, axis=0)
                db1 = np.sum(dZ1, axis=0)

                # Gradient update
                self.__W1 -= learning_rate * dW1
                self.__W2 -= learning_rate * dW2
                self.__b1 -= learning_rate * db1
                self.__b2 -= learning_rate * db2

    def predict(self, X):
        Z1 = ReLU(X @ self.__W1 + self.__b1)
        Z2 = Z1 @ self.__W2 + self.__b2

        return softmax(Z2)
