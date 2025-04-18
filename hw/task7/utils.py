import numpy as np
from dataclasses import astuple, dataclass


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


@dataclass
class BatchnormCache:
    X: np.ndarray
    mean: np.ndarray
    var: np.ndarray
    X_norm: np.ndarray
    gamma: float
    beta: float
    epsilon: float

    def __iter__(self):
        return iter(astuple(self))


def batchnorm_forward(X, gamma, beta, epsilon=1e-5):
    mean = np.mean(X, axis=0)  # мат. ожидание батча (μ)
    var = np.var(X, axis=0)  # дисперсия
    X_norm = (X - mean) / np.sqrt(var + epsilon)  # нормализация
    out = gamma * X_norm + beta  # сжатие и сдвиг

    return out, BatchnormCache(X, mean, var, X_norm, gamma, beta, epsilon)


def batchnorm_backward(dout, cache: BatchnormCache):
    X, mean, var, X_norm, gamma, beta, epsilon = cache
    m = X.shape[0]

    # градиент по нормализованному входу
    dX_norm = dout * gamma

    # градиенты по параметрам
    dgamma = np.sum(dout * X_norm, axis=0)
    dbeta = np.sum(dout, axis=0)

    # градиент по дисперсии
    dvar = np.sum(
        dX_norm * (X - mean) * (-1 / 2) * ((var + epsilon) ** (-3 / 2)), axis=0
    )
    # градиент по среднему
    dmean = np.sum(dX_norm * (-1 / np.sqrt(var + epsilon)), axis=0) + dvar * np.mean(
        -2 * (X - mean), axis=0
    )

    # градиент по входу X
    dX = (
        (dX_norm / np.sqrt(var + epsilon)) + ((dvar * 2 * (X - mean)) / m) + (dmean / m)
    )

    return dX, dgamma, dbeta


class MyClassifier:
    def __init__(self, input_num, classes_num, hidden_neurons_num):
        self.__W1 = 0.001 * np.random.randn(input_num, hidden_neurons_num)
        self.__W2 = 0.001 * np.random.randn(hidden_neurons_num, classes_num)

        self.__b1 = np.zeros(hidden_neurons_num)
        self.__b2 = np.zeros(classes_num)

    def fit(self, X, y, batch_size=100, learning_rate=1e-7, reg=1e-5, epochs=1):
        num_train = X.shape[0]
        loss_list = []

        for _ in range(epochs):
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

            loss_list.append(loss)
        return loss_list

    def predict(self, X):
        Z1 = ReLU(X @ self.__W1 + self.__b1)
        Z2 = Z1 @ self.__W2 + self.__b2

        return softmax(Z2)

    def predict_max(self, X):
        return np.argmax(self.predict(X), axis=1).astype(int)


class MyBatchnormClassifier:
    def __init__(self, input_num, classes_num, hidden_neurons_num):
        self.__W1 = 0.001 * np.random.randn(input_num, hidden_neurons_num)
        self.__W2 = 0.001 * np.random.randn(hidden_neurons_num, classes_num)

        self.__b1 = np.zeros(hidden_neurons_num)
        self.__b2 = np.zeros(classes_num)

        self.__gamma = np.ones(hidden_neurons_num)
        self.__beta = np.zeros(hidden_neurons_num)

    def fit(self, X, y, batch_size=100, learning_rate=1e-7, reg=1e-5, epochs=1):
        num_train = X.shape[0]
        loss_list = []

        for _ in range(epochs):
            shuffled_indices = np.arange(num_train)
            np.random.shuffle(shuffled_indices)

            sections = np.arange(batch_size, num_train, batch_size)
            batches_indices = np.array_split(shuffled_indices, sections)

            for batch_idx in batches_indices:
                batchX, batchY = X[batch_idx], y[batch_idx]

                # Forward pass
                H1 = batchX @ self.__W1 + self.__b1
                H1_norm, bn_cache = batchnorm_forward(H1, self.__gamma, self.__beta)

                Z1 = ReLU(H1_norm)
                Z2 = Z1 @ self.__W2 + self.__b2

                loss, dZ2 = softmax_with_cross_entropy(Z2, batchY)

                l2_loss1, l2_grad1 = l2_regularization(self.__W1, reg)
                l2_loss2, l2_grad2 = l2_regularization(self.__W2, reg)

                loss += l2_loss1 + l2_loss2

                # Backward pass

                dZ1 = dZ2 @ self.__W2.T
                dZ1 = ReLU_backward(dZ1, Z1)
                dH1, dgamma, dbeta = batchnorm_backward(dZ1, bn_cache)

                dW2 = (Z1.T @ dZ2) + l2_grad2
                dW1 = (batchX.T @ dH1) + l2_grad1

                db2 = np.sum(dZ2, axis=0)
                db1 = np.sum(dH1, axis=0)

                # Gradient update
                self.__W1 -= learning_rate * dW1
                self.__W2 -= learning_rate * dW2
                self.__b1 -= learning_rate * db1
                self.__b2 -= learning_rate * db2

                self.__gamma -= learning_rate * dgamma
                self.__beta -= learning_rate * dbeta

            loss_list.append(loss)
        return loss_list

    def predict(self, X):
        H1 = X @ self.__W1 + self.__b1
        H1_norm, _ = batchnorm_forward(H1, self.__gamma, self.__beta)

        Z1 = ReLU(H1_norm)
        Z2 = Z1 @ self.__W2 + self.__b2

        return softmax(Z2)

    def predict_max(self, X):
        return np.argmax(self.predict(X), axis=1).astype(int)


class MyBatchnormMomentumClassifier:
    def __init__(self, input_num, classes_num, hidden_neurons_num):
        self.__W1 = 0.001 * np.random.randn(input_num, hidden_neurons_num)
        self.__W2 = 0.001 * np.random.randn(hidden_neurons_num, classes_num)

        self.__b1 = np.zeros(hidden_neurons_num)
        self.__b2 = np.zeros(classes_num)

        self.__gamma = np.ones(hidden_neurons_num)
        self.__beta = np.zeros(hidden_neurons_num)

        self.__v_W1 = np.zeros_like(self.__W1)
        self.__v_W2 = np.zeros_like(self.__W2)
        self.__v_b1 = np.zeros_like(self.__b1)
        self.__v_b2 = np.zeros_like(self.__b2)
        self.__v_gamma = np.zeros_like(self.__gamma)
        self.__v_beta = np.zeros_like(self.__beta)

    def fit(self, X, y, batch_size=100, learning_rate=1e-7, reg=1e-5, epochs=1, mu=0.9):
        num_train = X.shape[0]
        loss_list = []

        for _ in range(epochs):
            shuffled_indices = np.arange(num_train)
            np.random.shuffle(shuffled_indices)

            sections = np.arange(batch_size, num_train, batch_size)
            batches_indices = np.array_split(shuffled_indices, sections)

            for batch_idx in batches_indices:
                batchX, batchY = X[batch_idx], y[batch_idx]

                # Forward pass
                H1 = batchX @ self.__W1 + self.__b1
                H1_norm, bn_cache = batchnorm_forward(H1, self.__gamma, self.__beta)

                Z1 = ReLU(H1_norm)
                Z2 = Z1 @ self.__W2 + self.__b2

                loss, dZ2 = softmax_with_cross_entropy(Z2, batchY)

                l2_loss1, l2_grad1 = l2_regularization(self.__W1, reg)
                l2_loss2, l2_grad2 = l2_regularization(self.__W2, reg)

                loss += l2_loss1 + l2_loss2

                # Backward pass

                dZ1 = dZ2 @ self.__W2.T
                dZ1 = ReLU_backward(dZ1, Z1)
                dH1, dgamma, dbeta = batchnorm_backward(dZ1, bn_cache)

                dW2 = (Z1.T @ dZ2) + l2_grad2
                dW1 = (batchX.T @ dH1) + l2_grad1

                db2 = np.sum(dZ2, axis=0)
                db1 = np.sum(dH1, axis=0)

                # Momentum update
                self.__v_W1 = mu * self.__v_W1 - learning_rate * dW1
                self.__v_W2 = mu * self.__v_W2 - learning_rate * dW2
                self.__v_b1 = mu * self.__v_b1 - learning_rate * db1
                self.__v_b2 = mu * self.__v_b2 - learning_rate * db2
                self.__v_gamma = mu * self.__v_gamma - learning_rate * dgamma
                self.__v_beta = mu * self.__v_beta - learning_rate * dbeta

                self.__W1 += self.__v_W1
                self.__W2 += self.__v_W2
                self.__b1 += self.__v_b1
                self.__b2 += self.__v_b2

                self.__gamma += self.__v_gamma
                self.__beta += self.__v_beta

            loss_list.append(loss)
        return loss_list

    def predict(self, X):
        H1 = X @ self.__W1 + self.__b1
        H1_norm, _ = batchnorm_forward(H1, self.__gamma, self.__beta)

        Z1 = ReLU(H1_norm)
        Z2 = Z1 @ self.__W2 + self.__b2

        return softmax(Z2)

    def predict_max(self, X):
        return np.argmax(self.predict(X), axis=1).astype(int)


class MyBatchnormAdamClassifier:
    def __init__(self, input_num, classes_num, hidden_neurons_num):
        self.__W1 = 0.001 * np.random.randn(input_num, hidden_neurons_num)
        self.__W2 = 0.001 * np.random.randn(hidden_neurons_num, classes_num)

        self.__b1 = np.zeros(hidden_neurons_num)
        self.__b2 = np.zeros(classes_num)

        self.__gamma = np.ones(hidden_neurons_num)
        self.__beta = np.zeros(hidden_neurons_num)

        self.__v_W1 = np.zeros_like(self.__W1)
        self.__v_W2 = np.zeros_like(self.__W2)
        self.__v_b1 = np.zeros_like(self.__b1)
        self.__v_b2 = np.zeros_like(self.__b2)
        self.__v_gamma = np.zeros_like(self.__gamma)
        self.__v_beta = np.zeros_like(self.__beta)

        self.__m_W1 = np.zeros_like(self.__W1)
        self.__m_W2 = np.zeros_like(self.__W2)
        self.__m_b1 = np.zeros_like(self.__b1)
        self.__m_b2 = np.zeros_like(self.__b2)
        self.__m_gamma = np.zeros_like(self.__gamma)
        self.__m_beta = np.zeros_like(self.__beta)

    def fit(
        self,
        X,
        y,
        batch_size=100,
        learning_rate=1e-7,
        reg=1e-5,
        epochs=1,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
    ):
        num_train = X.shape[0]
        loss_list = []

        for _ in range(epochs):
            shuffled_indices = np.arange(num_train)
            np.random.shuffle(shuffled_indices)

            sections = np.arange(batch_size, num_train, batch_size)
            batches_indices = np.array_split(shuffled_indices, sections)

            for batch_idx in batches_indices:
                batchX, batchY = X[batch_idx], y[batch_idx]

                # Forward pass
                H1 = batchX @ self.__W1 + self.__b1
                H1_norm, bn_cache = batchnorm_forward(H1, self.__gamma, self.__beta)

                Z1 = ReLU(H1_norm)
                Z2 = Z1 @ self.__W2 + self.__b2

                loss, dZ2 = softmax_with_cross_entropy(Z2, batchY)

                l2_loss1, l2_grad1 = l2_regularization(self.__W1, reg)
                l2_loss2, l2_grad2 = l2_regularization(self.__W2, reg)

                loss += l2_loss1 + l2_loss2

                # Backward pass

                dZ1 = dZ2 @ self.__W2.T
                dZ1 = ReLU_backward(dZ1, Z1)
                dH1, dgamma, dbeta = batchnorm_backward(dZ1, bn_cache)

                dW2 = (Z1.T @ dZ2) + l2_grad2
                dW1 = (batchX.T @ dH1) + l2_grad1

                db2 = np.sum(dZ2, axis=0)
                db1 = np.sum(dH1, axis=0)

                # Adam update
                self.__m_W1 = beta1 * self.__m_W1 + (1 - beta1) * dW1
                self.__m_W2 = beta1 * self.__m_W2 + (1 - beta1) * dW2
                self.__m_b1 = beta1 * self.__m_b1 + (1 - beta1) * db1
                self.__m_b2 = beta1 * self.__m_b2 + (1 - beta1) * db2
                self.__m_gamma = beta1 * self.__m_gamma + (1 - beta1) * dgamma
                self.__m_beta = beta1 * self.__m_beta + (1 - beta1) * dbeta

                self.__v_W1 = beta2 * self.__v_W1 + (1 - beta2) * (dW1**2)
                self.__v_W2 = beta2 * self.__v_W2 + (1 - beta2) * (dW2**2)
                self.__v_b1 = beta2 * self.__v_b1 + (1 - beta2) * (db1**2)
                self.__v_b2 = beta2 * self.__v_b2 + (1 - beta2) * (db2**2)
                self.__v_gamma = beta2 * self.__v_gamma + (1 - beta2) * (dgamma**2)
                self.__v_beta = beta2 * self.__v_beta + (1 - beta2) * (dbeta**2)

                self.__W1 += -learning_rate * self.__m_W1 / (np.sqrt(self.__v_W1) + eps)
                self.__W2 += -learning_rate * self.__m_W2 / (np.sqrt(self.__v_W2) + eps)
                self.__b1 += -learning_rate * self.__m_b1 / (np.sqrt(self.__v_b1) + eps)
                self.__b2 += -learning_rate * self.__m_b2 / (np.sqrt(self.__v_b2) + eps)

                self.__gamma += (
                    -learning_rate * self.__m_gamma / (np.sqrt(self.__v_gamma) + eps)
                )
                self.__beta += (
                    -learning_rate * self.__m_beta / (np.sqrt(self.__v_beta) + eps)
                )

            loss_list.append(loss)
        return loss_list

    def predict(self, X):
        H1 = X @ self.__W1 + self.__b1
        H1_norm, _ = batchnorm_forward(H1, self.__gamma, self.__beta)

        Z1 = ReLU(H1_norm)
        Z2 = Z1 @ self.__W2 + self.__b2

        return softmax(Z2)

    def predict_max(self, X):
        return np.argmax(self.predict(X), axis=1).astype(int)
