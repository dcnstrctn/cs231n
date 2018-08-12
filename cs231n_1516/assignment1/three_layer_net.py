
# coding: utf-8

# # Implementing a Neural Network
# In this exercise we will develop a neural network with fully-connected layers to perform classification, and test it out on the CIFAR-10 dataset.

# In[1]:


# A bit of setup

import numpy as np
import matplotlib.pyplot as plt

from cs231n.classifiers.neural_net3 import ThreeLayerNet

get_ipython().magic(u'matplotlib inline')
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# for auto-reloading external modules
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
get_ipython().magic(u'load_ext autoreload')
get_ipython().magic(u'autoreload 2')

def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


# We will use the class `TwoLayerNet` in the file `cs231n/classifiers/neural_net.py` to represent instances of our network. The network parameters are stored in the instance variable `self.params` where keys are string parameter names and values are numpy arrays. Below, we initialize toy data and a toy model that we will use to develop your implementation.



from cs231n.gradient_check import eval_numerical_gradient

# Use numeric gradient checking to check your implementation of the backward pass.
# If your implementation is correct, the difference between the numeric and
# analytic gradients should be less than 1e-8 for each of W1, W2, b1, and b2.

# In[11]:


from cs231n.data_utils import load_CIFAR10

def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for the two-layer neural net classifier. These are the same steps as
    we used for the SVM, but condensed to a single function.  
    """
    # Load the raw CIFAR-10 data
    cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
        
    # Subsample the data
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    # Reshape data to rows
    X_train = X_train.reshape(num_training, -1)
    X_val = X_val.reshape(num_validation, -1)
    X_test = X_test.reshape(num_test, -1)

    return X_train, y_train, X_val, y_val, X_test, y_test


# Invoke the above function to get our data.
X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()


# # Train a network
# To train our network we will use SGD with momentum. In addition, we will adjust the learning rate with an exponential learning rate schedule as optimization proceeds; after each epoch, we will reduce the learning rate by multiplying it by a decay rate.



# # Tune your hyperparameters
# 
# **What's wrong?**. Looking at the visualizations above, we see that the loss is decreasing more or less linearly, which seems to suggest that the learning rate may be too low. Moreover, there is no gap between the training and validation accuracy, suggesting that the model we used has low capacity, and that we should increase its size. On the other hand, with a very large model we would expect to see more overfitting, which would manifest itself as a very large gap between the training and validation accuracy.
# 
# **Tuning**. Tuning the hyperparameters and developing intuition for how they affect the final performance is a large part of using Neural Networks, so we want you to get a lot of practice. Below, you should experiment with different values of the various hyperparameters, including hidden layer size, learning rate, numer of training epochs, and regularization strength. You might also consider tuning the learning rate decay, but you should be able to get good performance using the default value.
# 
# **Approximate results**. You should be aim to achieve a classification accuracy of greater than 48% on the validation set. Our best network gets over 52% on the validation set.
# 
# **Experiment**: You goal in this exercise is to get as good of a result on CIFAR-10 as you can, with a fully-connected Neural Network. For every 1% above 52% on the Test set we will award you with one extra bonus point. Feel free implement your own techniques (e.g. PCA to reduce dimensionality, or adding dropout, or adding features to the solver, etc.).

# In[27]:


best_net = None # store the best model into this 

#################################################################################
# TODO: Tune hyperparameters using the validation set. Store your best trained  #
# model in best_net.                                                            #
#                                                                               #
# To help debug your network, it may help to use visualizations similar to the  #
# ones we used above; these visualizations will have significant qualitative    #
# differences from the ones we saw above for the poorly tuned network.          #
#                                                                               #
# Tweaking hyperparameters by hand can be fun, but you might find it useful to  #
# write code to sweep through possible combinations of hyperparameters          #
# automatically like we did on the previous exercises.                          #
#################################################################################
#pass
results = {}
best_val = -1
best_softmax = None
learning_rates = [1.3e-3]
regularization_strengths = [0.8]
input_size = 32 * 32 * 3
hidden_size1 = [250]
hidden_size2 = 100
num_classes = 10

for lr in learning_rates:
    for rs in regularization_strengths:
        for hs in hidden_size1:
            net = ThreeLayerNet(input_size, hs, hidden_size2, num_classes)
            stats = net.train(X_train, y_train, X_val, y_val,
                num_iters=1000, batch_size=300,
                learning_rate=lr, learning_rate_decay=0.95, reg=rs)
            training_accuracy = np.mean(y_train == net.predict(X_train))
            validation_accuracy = np.mean(y_val == net.predict(X_val))
            results[(lr, rs, hs)] = [training_accuracy, validation_accuracy]
            if validation_accuracy > best_val:
                best_val = validation_accuracy
                best_net = net
            # Plot the loss function and train / validation accuracies
            #print 'lr %e reg %e hs %e'% (lr, reg, size)
            #plt.subplot(2, 1, 1)
            #plt.plot(stats['loss_history'])
            #plt.title('Loss history')
            #plt.xlabel('Iteration')
            #plt.ylabel('Loss')

            #plt.subplot(2, 1, 2)
            #plt.plot(stats['train_acc_history'], label='train')
            #plt.plot(stats['val_acc_history'], label='val')
            #plt.title('Classification accuracy history')
            #plt.xlabel('Epoch')
            #plt.ylabel('Clasification accuracy')
            #plt.show()
            
for lr, reg, size in sorted(results):
    train_accuracy, val_accuracy = results[(lr, reg, size)]
    print 'lr %e reg %e hs %e train accuracy: %f val accuracy: %f' % (
                lr, reg, size, train_accuracy, val_accuracy)
    
print 'best validation accuracy achieved: %f' % best_val
#################################################################################
#                               END OF YOUR CODE                                #
#################################################################################


# In[ ]:


# visualize the weights of the best network
show_net_weights(best_net)


# # Run on the test set
# When you are done experimenting, you should evaluate your final trained network on the test set; you should get above 48%.
# 
# **We will give you extra bonus point for every 1% of accuracy above 52%.**

# In[ ]:



