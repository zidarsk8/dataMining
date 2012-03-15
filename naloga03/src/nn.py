from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer
from pybrain.structure import TanhLayer
from pybrain.structure import FullConnection
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
from time import time
import os
import pickle




arrFile = 'minidata/nnArr.pickled'
tesFile = 'minidata/nnTes.pickled'
labFile = 'minidata/nnLab.pickled'
if False and os.path.isfile(arrFile) and os.path.isfile(tesFile) and os.path.isfile(labFile):
	print "reading labels from pickle file"
	arr = pickle.load(open(arrFile))
	tes = pickle.load(open(tesFile))
	lab = pickle.load(open(labFile))
else:
	f = open("minidata/trainingData.csv")
	arr = [[int(y) for y in x.strip().split("\t")] for x in f.readlines()]
	f = open("minidata/trainingData.csv")
	tes = [[int(y) for y in x.strip().split("\t")] for x in f.readlines()]
	
	aa, tt, rind = zip(*arr), zip(*tes), []
	[rind.append(j) for j,i in enumerate(aa) if i.count(0) > len(i)-20]
	[aa.pop(i) for i in sorted(rind,reverse=True)]
	[tt.pop(i) for i in sorted(rind,reverse=True)]
	arr, tes = zip(*aa), zip(*tt)
	
	tl = open("minidata/trainingLabels.csv")
	ll = [[int(j) for j in i.strip().split(",")] for i in tl.readlines()]
	m = max(map(max,ll))
	lab = [[1 if j in x else 0 for j in xrange(m+1) ] for x in ll]
	
	pickle.dump(arr,file(arrFile,"w"),-1)
	pickle.dump(tes,file(tesFile,"w"),-1)
	pickle.dump(lab,file(labFile,"w"),-1)


	

inLen = len(arr[0])
outLen = len(lab[0])

print "short len : ",inLen

print "loaded data now for calculating"



n = FeedForwardNetwork()

inLayer = LinearLayer(inLen)
hiddenLayer = SigmoidLayer(inLen*2)
outLayer = LinearLayer(outLen)

n.addInputModule(inLayer)
n.addModule(hiddenLayer)
n.addOutputModule(outLayer)

in_to_hidden = FullConnection(inLayer, hiddenLayer)
hidden_to_out = FullConnection(hiddenLayer, outLayer)

n.addConnection(in_to_hidden)
n.addConnection(hidden_to_out)
n.sortModules()

print "made network"
#print n.activate(arr[0])

ds = SupervisedDataSet(inLen, outLen)

for i in xrange(len(arr)):
	ds.addSample(arr[i],lab[i])

print "filled training dataset"

trainer = BackpropTrainer(n, ds)

trainer.train()
print "training done1"
trainer.train()
print "training done2"
trainer.train()
print "training done3"
trainer.train()
print "training done4"

result = []
def sk(x): return x[1]
for i,t in enumerate(tes):
	print i
	r = [(i,j) for i,j in enumerate(n.activate(t))]

	result.append([i for i,j in sorted(r, key=sk, reverse=True)])


f = file("nn%d.csv" % time(),"w")
f.write("\n".join([",".join([str(x) for x in i]) for i in result ]))
f.flush()
f.close()
