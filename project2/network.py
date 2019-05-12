import numpy as np
import matplotlib.pyplot as plt

def init_two_layer_model(input_size, hidden_size, output_size):
  # initialize a model
  model = {}
  model['W1'] = 0.00001 * np.random.randn(input_size, hidden_size)
  model['b1'] = np.zeros(hidden_size)
  model['W2'] = 0.00001 * np.random.randn(hidden_size, output_size)
  model['b2'] = np.zeros(output_size)
  return model


def two_layer_net(X, model, y=None, reg=0.0):
    # unpack variables from the model dictionary
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    N, D = X.shape

    # compute the forward pass
    scores = None
    hidden_layer = np.maximum(0, np.dot(X, W1) + b1)  # ReLU activation
    scores = np.dot(hidden_layer, W2) + b2

    # If the targets are not given then jump out, we're done
    if y is None:
        return scores

    # compute the loss
    loss = None
    # compute the class probabilities
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)  # [N x K]

    # average cross-entropy loss and regularization
    corect_logprobs = -np.log(probs[range(N), y])
    data_loss = np.sum(corect_logprobs) / N
    reg_loss = 0.5 * reg * np.sum(W1 * W1) + 0.5 * reg * np.sum(W2 * W2)
    loss = data_loss + reg_loss


    # compute the gradients
    grads = {}

    # compute the gradient on scores
    dscores = probs
    dscores[range(N), y] -= 1
    dscores /= N

    # W2 and b2
    grads['W2'] = np.dot(hidden_layer.T, dscores)
    grads['b2'] = np.sum(dscores, axis=0)
    # next backprop into hidden layer
    dhidden = np.dot(dscores, W2.T)
    # backprop the ReLU non-linearity
    dhidden[hidden_layer <= 0] = 0
    # finally into W,b
    grads['W1'] = np.dot(X.T, dhidden)
    grads['b1'] = np.sum(dhidden, axis=0)

    # add regularization gradient contribution
    grads['W2'] += reg * W2
    grads['W1'] += reg * W1


    return loss, grads