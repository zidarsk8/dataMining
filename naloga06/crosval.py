import matplotlib.pyplot as plot
import numpy as np
import Orange
import random

data = Orange.data.Table("data/train.tab");

X, y, _ = data.to_numpy();

# m = rows, n = columns
m,n = X.shape 

folds = 10;

random.seed(12345)

index = 3