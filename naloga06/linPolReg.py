import numpy as np
import matplotlib.pyplot as plt

from scipy import optimize

# logReg.py naj bo datoteka za vaso domaco nalogo, kjer imate funkcijo ki je v predlogi.
import logReg
reload(logReg)

np.random.seed(42)

def construct_log_reg_data(m=20,border=1.0):
    X = np.random.rand(m,1)
    y = X.T[0] < np.random.rand(m)*border
    return X,y
    
def normal_lin_reg(X, y, lambda_=0.):
    """Parameters of linear regression by normal equation."""
    n = X.shape[1] - 1
    return np.linalg.inv((X.T.dot(X)) + lambda_*np.diag(np.append(np.zeros(1), np.ones(n)))).dot(X.T.dot(y))

def predict_lin_reg(X, theta):
    """Linear regression predictions for a data set"""
    return X.dot(theta)

def sigmoid(z):
    """Sigmoid (logistic) function."""
    return 1. / (1. + np.exp(-z))

def predict_log_reg(X, theta):
    """Predict class value based on linear regression model."""
    return sigmoid(X.dot(theta))

def feature_extend(X, n_features=1):
    """Return data matrix with added columns of x1**n"""
    return np.column_stack([X] + [X[:,1]**(i+2) for i in range(n_features)])


def log_reg(X, y, lambda_):
    m = X.shape[0]
    alpha = 0.001
    rate = alpha / m
    thetas0 = np.zeros(X.shape[1])
    sigmoid = lambda z: 1 / (1 + np.exp(-z))
     
    def cost(thetas, X, y, lambda_):
        h = sigmoid(X.dot(thetas))
        return (-sum(y * np.log(h) + (1 - y) * np.log(1 - h)) + lambda_ * sum(thetas**2)) / m
     
    def grad(thetas, X, y, lambda_):
        return thetas * (1 - lambda_ * rate) + rate * (sigmoid(X.dot(thetas)) - y).dot(X)
       
    return optimize.fmin_l_bfgs_b(cost, thetas0, grad, args=(X, y, lambda_))


nf = 2 # number of higher-order features
k = 100 # number of test examples
m = 19 # train data set size
lambda_ = 0.005 # regularization, increase to 0.001 to see the effect


X, y = construct_log_reg_data(m=m)
X = np.column_stack([np.ones((X.shape[0],1)),X]) #dodamo enke na zacetek za theta0


X = feature_extend(X, n_features=nf)
T = np.array([np.ones(k)] + [np.linspace(min(X[:,1]), max(X[:,1]), k)]).T
T = feature_extend(T, n_features=nf)


thetaLin = normal_lin_reg(X, y, lambda_=lambda_)
thetaLog, _, _ = logReg.log_reg(X, y, lambda_)


ylin = predict_lin_reg(T, thetaLin)
ylog = predict_log_reg(T, thetaLog)

print "Normal eq. lin:", thetaLin
print "Normal eq. log:", thetaLog

plt.close()
plt.plot(X[:,1], y, "o")
#plt.plot(T[:,1], ylin)
for i in range(4):
    plt.plot(T[:,1], logReg.predict_log_reg(X, y, T, {"lambda_" : 0.001*i**2} ))
plt.xlabel("x1")
plt.ylabel("y")
plt.title("Polynomial regression (lambda=%.5f)" % lambda_)
plt.savefig("linlogreg.png")
plt.show()