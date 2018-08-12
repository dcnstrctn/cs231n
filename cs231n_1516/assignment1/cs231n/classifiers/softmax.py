import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  #pass
  num_train = X.shape[0]
  num_classes = W.shape[1]
  
  for i in range(num_train):#X[i] (1, D) --(1, 1)   
      scores = X[i].dot(W)
      shift = max(scores)
      scores -= shift #(1, C)
      loss += -scores[y[i]] + np.log(sum(np.exp(scores)))
      
      for c in range(num_classes):
          e = np.exp(scores[c])/sum(np.exp(scores)) #(1, 1)
          if c == y[i]:
              dW[:, c] += e*(X[i].T) - X[i].T #(D, 1)
          else:
              dW[:, c] += e*(X[i].T)
              
  loss = loss/num_train
  loss += 0.5*reg*np.sum(W*W)
  dW /= num_train
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  #pass
  num_train = X.shape[0]

  scores = X.dot(W) #(N, C)
  scores -= np.max(scores, axis = 1).reshape(-1, 1)
  probabilities = np.exp(scores)/np.sum(np.exp(scores), axis = 1).reshape(-1, 1)#(N, C) + (N, 1) broadcast
  correct_prob = probabilities[range(num_train), y]
  loss = sum(-np.log(correct_prob))/num_train + 0.5*reg*np.sum(W*W)
  
  probabilities[range(num_train), y] -= 1
  dW = (X.T).dot(probabilities)/num_train
  dW += reg*W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

