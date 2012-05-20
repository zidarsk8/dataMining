from scipy import optimize
import numpy as np
#import Orange


def sigmoid(z):
    """Sigmoid (logistic) function."""
    return 1. / (1. + np.exp(-z))


def construct_data(m=10, n=1, sigma=0.5, f=None):
    """Return data with m attributes and continuous class."""
    X = np.array([np.ones(m)] + [np.random.random(m) for _ in range(n)]).T
    if not f:
        theta = np.arange(n+1)
        y = X.dot(theta)
    else:
        y = np.array([f(x[1:]) for x in X])
    return X, y + np.random.normal(0, sigma, m)


def feature_extend(X,nonbin=[], n_features=1):
    """Return data matrix with added columns of x1**n"""
    if type(nonbin) != list or nonbin == []:
        nonbin = range(X.shape[1])
    return np.column_stack([X] + [X[:,nonbin]**(i+2) for i in range(n_features)])


def log_reg(X, y, lambda_):
    
    def cost(thetas, X, y, lambda_):
        return -sum(y * np.log(sigmoid(X.dot(thetas))) + (1 - y) * np.log(1 - sigmoid(X.dot(thetas))))/2 + lambda_ * sum(thetas**2)
        
    def grad(thetas, X, y, lambda_):
        return (sigmoid(X.dot(thetas))-y).dot(X)/X.shape[1] + lambda_ * sum(thetas**2)
    
    return optimize.fmin_l_bfgs_b(cost, np.ones(X.shape[1]), grad, args=(X, y, lambda_))


def predict_log_reg(X , y, testX,  args={}):
    """Predict class value based on linear regression model."""
    
    lambda_ = args["lambda_"] if type(args) == dict and args.has_key("lambda_") else 0.000
    #n_features = args["n_features"] if type(args) == dict and args.has_key("n_features") else 0
    
    
    #nonbin = [i for i in xrange(X.shape[1]) if np.unique(X[:,i]).size > 2 ]
    #X = feature_extend(X, nonbin, n_features);
    #X = np.column_stack([np.ones((X.shape[0],1)),X]) #dodamo enke na zacetek za theta0
    
    thetas, _, _ = log_reg(X, y, lambda_)
    
    return sigmoid(testX.dot(thetas))
    

if __name__ == "__main__":
    print "loading data"
    data = Orange.data.Table("data/train.tab")
    X,y,_ = data.to_numpy()
    
    n_features = 5 # number of higher-order features
    k = 50 # number of test examples
    m = 10 # train data set size
    lambda_ = 0.000 # regularization, increase to 0.001 to see the effect
    
    
    nonbin = [i for i in xrange(X.shape[1]) if np.unique(X[:,i]).size > 2 ]
    X = feature_extend(X, nonbin, n_features);
    X = np.column_stack([np.ones((X.shape[0],1)),X]) #dodamo enke na zacetek za theta0
        
    x, f, d = log_reg(X, y, lambda_)

    

#from scipy.optimize.optimize import fmin_cg, fmin_bfgs, fmin
#import numpy as np
#
#def sigmoid(x):
#    return 1.0 / (1.0 + np.exp(-x))
#
#class SyntheticClassifierData():
#    
#    def __init__(self, N, d):
#        """ Create N instances of d dimensional input vectors and a 1D
#        class label (-1 or 1). """
#        
#        means = .05 * np.random.randn(2, d)
#        
#        self.X_train = np.zeros((N, d))
#        self.Y_train = np.zeros(N)        
#        for i in range(N):
#            if np.random.random() > .5:
#                y = 1
#            else:
#                y = 0
#            self.X_train[i, :] = np.random.random(d) + means[y, :]
#            self.Y_train[i] = 2.0 * y - 1
#        
#        self.X_test = np.zeros((N, d))
#        self.Y_test = np.zeros(N)        
#        for i in range(N):
#            if np.random.randn() > .5:
#                y = 1
#            else:
#                y = 0
#            self.X_test[i, :] = np.random.random(d) + means[y, :]
#            self.Y_test[i] = 2.0 * y - 1
#
#
#class LogisticRegression():
#    """ A simple logistic regression model with L2 regularization (zero-mean
#    Gaussian priors on parameters). """
#
#    def __init__(self, x_train=None, y_train=None, x_test=None, y_test=None,
#                 alpha=.1, synthetic=False):
#        
#        # Set L2 regularization strength
#        self.alpha = alpha
#        
#        # Set the data.
#        self.set_data(x_train, y_train, x_test, y_test)
#        
#        # Initialize parameters to zero, for lack of a better choice.
#        self.betas = np.zeros(self.x_train.shape[1])
#
#
#    def negative_lik(self, betas):
#        return -1 * self.lik(betas)
#
#
#    def lik(self, betas):
#        """ Likelihood of the data under the current settings of parameters. """
#        
#        # Data likelihood
#        l = 0
#        for i in range(self.n):
#            l += log(sigmoid(self.y_train[i] * \
#                             np.dot(betas, self.x_train[i,:])))
#        
#        # Prior likelihood
#        for k in range(1, self.x_train.shape[1]):
#            l -= (self.alpha / 2.0) * self.betas[k]**2
#            
#        return l
#
#
#    def train(self):
#        """ Define the gradient and hand it off to a scipy gradient-based
#        optimizer. """
#        
#        # Define the derivative of the likelihood with respect to beta_k.
#        # Need to multiply by -1 because we will be minimizing.
#        dB_k = lambda B, k : (k > 0) * self.alpha * B[k] - np.sum([ \
#                                     self.y_train[i] * self.x_train[i, k] * \
#                                     sigmoid(-self.y_train[i] *\
#                                             np.dot(B, self.x_train[i,:])) \
#                                     for i in range(self.n)])
#        
#        # The full gradient is just an array of componentwise derivatives
#        dB = lambda B : np.array([dB_k(B, k) \
#                                  for k in range(self.x_train.shape[1])])
#        
#        # Optimize
#        self.betas = fmin_bfgs(self.negative_lik, self.betas, fprime=dB)
#
#
#    def set_data(self, x_train, y_train, x_test, y_test):
#        """ Take data that's already been generated. """
#        
#        self.x_train = x_train
#        self.y_train = y_train
#        self.x_test = x_test
#        self.y_test = y_test
#        self.n = y_train.shape[0]
#
#
#    def training_reconstruction(self):
#        p_y1 = np.zeros(self.n)
#        for i in range(self.n):
#            p_y1[i] = sigmoid(np.dot(self.betas, self.x_train[i,:]))
#        
#        return p_y1
#
#
#    def test_predictions(self):
#        p_y1 = np.zeros(self.n)
#        for i in range(self.n):
#            p_y1[i] = sigmoid(np.dot(self.betas, self.x_test[i,:]))
#        
#        return p_y1
#    
#    
#    def plot_training_reconstruction(self):
#        plot(np.arange(self.n), .5 + .5 * self.y_train, 'bo')
#        plot(np.arange(self.n), self.training_reconstruction(), 'rx')
#        ylim([-.1, 1.1])
#
#
#    def plot_test_predictions(self):
#        plot(np.arange(self.n), .5 + .5 * self.y_test, 'yo')
#        plot(np.arange(self.n), self.test_predictions(), 'rx')
#        ylim([-.1, 1.1])
#
#
#
#if __name__ == "__main__":
#    from pylab import *
#
#    # Create 20 dimensional data set with 25 points -- this will be
#    # susceptible to overfitting.
#    data = SyntheticClassifierData(25, 20)
#
#    # Run for a variety of regularization strengths
#    alphas = [0, .001, .01, .1]
#    for j, a in enumerate(alphas):
#       
#        # Create a new learner, but use the same data for each run
#        lr = LogisticRegression(x_train=data.X_train, y_train=data.Y_train,
#                                x_test=data.X_test, y_test=data.Y_test,
#                                alpha=a)
#        
#        print "Initial likelihood:"
#        print lr.lik(lr.betas)
#        
#        # Train the model
#        lr.train()
#        
#        # Display execution info
#        print "Final betas:"
#        print lr.betas
#        print "Final lik:"
#        print lr.lik(lr.betas)
#        
#        # Plot the results
#        subplot(len(alphas), 2, 2*j + 1)
#        lr.plot_training_reconstruction()
#        ylabel("Alpha=%s" % a)
#        if j == 0:
#            title("Training set reconstructions")
#        
#        subplot(len(alphas), 2, 2*j + 2)
#        lr.plot_test_predictions()
#        if j == 0:
#            title("Test set predictions")
#
#    show()
