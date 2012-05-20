import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import meshgrid
import Orange

def feature_extend(X):
    """Extend design matrix X with higher-order features."""
    return np.column_stack([X] + [X[:,1]*X[:,2]] + [X[:,1]**2] + [X[:,2]**2])

def batch_log_reg(X, y, theta, alpha=0.001, eps=1e-5, max_iterations=100, reg_lambda=0):
    """Return parameters of logistic regression model computed by batch gradient descent."""
    J = []
    m = X.shape[1]
    rate = alpha / m
    reg = (1 - rate * reg_lambda) * np.ones(theta.size)
    reg[0] = 1.
    for i in range(max_iterations):
        J.append(-sum(y * np.log(sigmoid(X.dot(theta))) + (1 - y) * np.log(1 - sigmoid(X.dot(theta))))/m)
        if eps and len(J)>2 and (J[-2] - J[-1] < eps):
            print "Break at", i
            break
        theta = reg * theta - rate * (sigmoid(X.dot(theta))-y).dot(X)/m
    return theta, J

def sigmoid(z):
    """Sigmoid (logistic) function."""
    return 1. / (1. + np.exp(-z))

def predict_log_reg(X, theta):
    """Predict class value based on linear regression model."""
    return sigmoid(X.dot(theta))

def test_mesh(x, y, theta):
    """Return probabilities for (x,y) test mesh."""
    T = np.column_stack([np.ones(x.size), x.reshape(x.size), y.reshape(y.size)])
    T = feature_extend(T)
    return predict_log_reg(T, theta).reshape(x.shape)


if __name__ == "__main__":
    data = Orange.data.Table("data/data-quad.tab")
    X, y, _ = data.to_numpy()
    m = X.shape[0]
    n = X.shape[1]
    X = np.column_stack([np.ones(m), X])
    X = feature_extend(X)
    
    theta, j_hist = batch_log_reg(X, y, alpha=0.1, theta=np.zeros(X.shape[1]), max_iterations=10000)
    print "Grad desc:", theta
    
    plt.close()
    grid = 30
    x1 = np.linspace(min(X[:,1]), max(X[:,1]), grid)
    x2 = np.linspace(min(X[:,2]), max(X[:,2]), grid)
    X1, X2 = meshgrid(x1, x2)
    
    Z = test_mesh(X1, X2, theta)
    plt.pcolor(X1, X2, Z) # plt.cm.gray
    
    plt.scatter(X[:,1], X[:,2], c=["y" if yi else "b" for yi in y])
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Logistic regression with non-linear decision boundary")
    plt.savefig("0.pdf")
