from multiprocessing import Pool
import pickle
import random



print "loading data"
data = pickle.load(file("data/minidata400x200.pkl"))

X, y, _ = data.to_numpy()
m,n = X.shape 
folds = 5
trees = 20
method = "knn"

m = len(data)

indexes = [int(float(i)/m*folds) for i in range(m)]
random.shuffle(indexes)
yPred = list(indexes)

def cv(fold):
    trainD = data.select(indexes,fold,negate=1)
    testD = data.select(indexes,fold)
    print X.size
    return fold*2

p = Pool(processes=2)

rr = p.map(cv, range(folds))

print rr



