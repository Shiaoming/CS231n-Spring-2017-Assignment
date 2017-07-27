import numpy as np
from random import shuffle
#from past.builtins import xrange

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
  train_num = X.shape[0]
  num_classes = np.max(y) + 1
  for i in xrange(train_num):
      y_pred = X[i,:].dot(W)
      y_pred = y_pred - np.max(y_pred)
      y_label = y[i]      
      prob = np.exp(y_pred[y_label]) / np.sum(np.exp(y_pred))
      loss += -1*np.log(prob)
      for j in xrange(num_classes):
          if j != y[i]:
              dW[:,j] += np.exp(y_pred[j])/np.sum(np.exp(y_pred)) * X[i]
          else:
              dW[:,y[i]] += (-1 + np.exp(y_pred[j])/np.sum(np.exp(y_pred))) * X[i]
              
  loss /= train_num
  dW /= train_num
  loss += 0.5*reg*np.sum(np.square(W))
  dW += reg*W
  
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
  num_train , dims = X.shape
  
  scores = np.dot(X, W)# N by C
  scores -= np.max(scores,axis=1,keepdims=True) 
  expscores = np.exp(scores)
  p = expscores / np.sum(expscores,axis=1,keepdims=True)
  y_trueClass = np.zeros_like(p)
  y_trueClass[range(num_train),y] = 1.0
  
  loss = -1 * np.sum(y_trueClass*np.log(p)) / num_train + 0.5*reg*np.sum(np.square(W))

  
  dW = np.dot(X.transpose(), p - y_trueClass)
  dW /= num_train
  dW += reg * W
#  scores = X.dot(W)        # N by C
#  num_train = X.shape[0]
#  num_classes = W.shape[1]
#  scores_correct = scores[np.arange(num_train), y]   # 1 by N
#  scores_correct = np.reshape(scores_correct, (num_train, 1))  # N by 1
#  margins = scores - scores_correct + 1.0     # N by C
#  margins[np.arange(num_train), y] = 0.0
#  margins[margins <= 0] = 0.0
#  loss += np.sum(margins) / num_train
#  loss += 0.5 * reg * np.sum(W * W)
#  # compute the gradient
#  margins[margins > 0] = 1.0
#  row_sum = np.sum(margins, axis=1)                  # 1 by N
#  margins[np.arange(num_train), y] = -row_sum        
#  dW += np.dot(X.T, margins)/num_train + reg * W     # D by C
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

