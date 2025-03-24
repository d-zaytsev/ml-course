import numpy as np
from abc import ABC, abstractmethod


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


def linear_softmax(X, W, target_index):
    """
    Performs linear classification and returns loss and gradient over W

    Arguments:
      X, np array, shape (num_batch, num_features) - batch of images
      W, np array, shape (num_features, classes) - weights
      target_index, np array, shape (num_batch) - index of target classes

    Returns:
      loss, single value - cross-entropy loss
      gradient, np.array same shape as W - gradient of weight by loss

    """
    predictions = np.dot(X, W)

    loss, dl_dpred = softmax_with_cross_entropy(
        predictions, target_index
    )  # dpred (batch x classes)
    dpred_dw = np.transpose(X)  # dw (features x batch)
    dW = np.dot(dpred_dw, dl_dpred)

    return loss, dW


class NeuronLayer(ABC):
    """Abstract base class for a neural network layer."""

    @abstractmethod
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, d_output: np.ndarray, learning_rate: float) -> np.ndarray:
        """Computes the backward pass.

        Args:
            d_output (np.ndarray): Gradient of the loss with respect to the output.
            learning_rate (float): Learning rate for parameter updates.

        Returns:
            np.ndarray: Gradient of the loss with respect to the input.
        """
        pass


class LinearLayer(NeuronLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.__W = np.random.randn(input_size, output_size) * 0.001
        self.__b = np.zeros((1, output_size))  # ???

    def forward(self, inputs):
        self.__input = inputs
        self.__output = (inputs @ self.__W) + self.__b

        return self.__output

    def backward(self, d_output, learning_rate):
        d_input = d_output @ self.__W.T
        d_weights = self.__input.T @ d_output
        d_biases = np.sum(d_output, axis=0, keepdims=True)

        self.__W -= learning_rate * d_weights
        self.__b -= learning_rate * d_biases

        return d_input


class ReLULayer(NeuronLayer):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        self.__input = inputs
        self.__output = np.maximum(0, inputs)

        return self.__output

    def backward(self, d_output, learning_rate):
        d_input = d_output * (self.input > 0)
        return d_input


class LinearSoftmaxClassifier:
    def __init__(self, num_hidden_neurons):
        self.__num_hidden_neurons = num_hidden_neurons
        self.__W1 = None
        self.__W2 = None

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
        num_input_neurons = X.shape[1]
        num_output_neurons = np.max(y) + 1  # Classes

        if self.__W1 is None:
            assert self.__W2 is None
            # input x hidden neurons number
            self.__W1 = 0.001 * np.random.randn(
                num_input_neurons, self.__num_hidden_neurons
            )
            # hidden neurons number x output
            self.__W2 = 0.001 * np.random.randn(
                self.__num_hidden_neurons, num_output_neurons
            )

        for epoch in range(epochs):
            shuffled_indices = np.arange(num_train)
            np.random.shuffle(shuffled_indices)

            sections = np.arange(batch_size, num_train, batch_size)
            batches_indices = np.array_split(shuffled_indices, sections)

            for batch_idx in batches_indices:
                batchX, batchY = X[batch_idx], y[batch_idx]

                Z1 = batchX @ self.__W1
                A1 = ReLU(Z1)
                Z2 = A1 @ self.__W2

                A2 = softmax(Z2)  # probabilities
                loss = cross_entropy_loss(A2, batchY)

                # TODO backward pass

                # loss, grad = linear_softmax(batchX, self.__W1, batchY)

                # l2_loss, l2_grad = l2_regularization(self.__W1, reg)
                # loss += l2_loss
                # grad += l2_grad

                # self.__W1 -= learning_rate * grad

            print("Epoch %i, loss: %f" % (epoch, loss))

    def predict(self, X):
        Z1 = X @ self.__W1
        A1 = ReLU(Z1)
        Z2 = A1 @ self.__W2
        A2 = softmax(Z2)

        return A2
