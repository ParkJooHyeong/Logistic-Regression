# Gradient Descent and Logistic Regression

import numpy as np


# logistic function
def logistic(z):
    return 1/(1+np.exp(-z))


# Initialize weights, bias
def initialize_with_zeros(dim):
    w = np.zeros(shape=(dim, 1))
    b = 0
    assert (w.shape == (dim, 1))
    assert (isinstance(b, float) or isinstance(b, int))
    return w, b


def propagate(w, b, X, Y):
    # eps : Avoid -inf of log operation
    eps=1e-5
    m = X.shape[0]

    # Forward propagation
    Y_hat = logistic(np.dot(np.transpose(w),np.transpose(X)) + b)
    # calculate cost using binary cross-entropy with regularizer
    cost = -1 / m * np.sum(Y * np.log(Y_hat+eps) + (1 - Y) * np.log(1 - Y_hat + eps)) + 0.001/2/m*(np.sum(w**2)+b**2)

    # Backward propagation
    dw = 1 / m * np.dot(np.transpose(X), np.transpose(Y_hat - Y))
    db = 1 / m * np.sum(Y_hat - Y)

    assert (dw.shape == w.shape)
    assert (db.dtype == float)
    cost = np.squeeze(cost)
    assert (cost.shape == ())
    grads = {"dw": dw, "db": db}
    return grads, cost


# Optimize : update new parameters using Gradient Descent
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    costs = []

    for i in range(num_iterations):

        # Cost and gradient calculation
        grads, cost = propagate(w, b, X, Y)

        dw = grads["dw"]
        db = grads["db"]

        w = w - learning_rate * dw
        b = b - learning_rate * db

        costs.append(cost)

        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    params = {"w": w,
              "b": b}

    grads = {"dw": dw,
             "db": db}

    return params, grads, costs


def model(X, Y, num_iterations=2000, learning_rate=0.05, print_cost=False):

    w, b = initialize_with_zeros(X.shape[1])

    parameters, grads, costs = optimize(w, b, X, Y, num_iterations, learning_rate, print_cost)

    w = parameters["w"]
    b = parameters["b"]

    d = {"costs": costs,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}

    return d


# calculate Conversion probability
def predict_prob(model, X):
    return logistic(np.dot(np.transpose(model["w"]),X.T)+model["b"])


# predict label
def predict(model, X):
    return np.round(logistic(np.dot(np.transpose(model["w"]), X.T) + model["b"]))


# calculate accuracy
def score(model, X, y):
    total= X.shape[0]
    target = predict(model,X)
    result = target == y
    ans = np.unique(result, return_counts=True)[1]
    return ans[1]/total


# save model, file name : model.dat
def saveModel(model):
    np.savetxt('model/model.dat',[model['costs'][-1],model['w'][0], model['w'][1], model['b'],
                model['learning_rate'], model['num_iterations']])


def loadModel():
    target = np.loadtxt('model/model.dat')
    result = {
        "costs": target[0],
        "w": np.array([[target[1]],[target[2]]]),
        "b": target[3],
        "learning_rate": target[4],
        "num_iterations": target[5]
    }
    return result
