import matplotlib.pyplot as plot
import numpy as np
import Orange


data = Orange.data.Table("data/train.tab")


trainD, y, _ = data.to_numpy()

print "stevilo primerov(m) %d\nstevilo atributov(n) %d" % trainD.shape

print "stevilo napovedanih razredov %d/%d  oziroma %.1f%%" %\
		(sum(y),len(y),(float(sum(y))/len(y)*100.0))

m,n = trainD.shape
print "stevilo nul v celotni matriki %d/%d oziroma %.1f%%" %\
		(sum(sum(trainD==0)),(n*m) , sum(sum(trainD==0))*100.0/(m*n))

p = True

nenicelnih = sum(trainD != 0)
if p:
	plot.hist(nenicelnih,bins=50)
	plot.xlabel("st. nenicelnih vrednosti")
	plot.ylabel("st. atributov")
	plot.show()

razlicnih = np.asarray([np.unique(trainD[:,i]).size for i in xrange(n)])
if p:
	plot.hist(razlicnih,bins=50,log=True)
	plot.xlabel("stevilo razlicnih vrednosti")
	plot.ylabel("st. atributov")
	plot.show()


razlicnihProcent = (razlicnih-1+(nenicelnih==m))*100.0/nenicelnih
if p:
	plot.hist(razlicnihProcent,bins=50,log=True)
	plot.xlabel("stevilo razlicnih v odvisnosti od stevila nenicelnih")
	plot.ylabel("st. atributov")
	plot.show()


maxStevilo = [x.max() for x in trainD.T]
if p:
	plot.hist(maxStevilo,bins=50)
	plot.xlabel("maksimalno stevilo v stolpcu")
	plot.ylabel("st. atributov")
	plot.show()


if not p:
	means = [a.mean() for a in trainD.T]
	for i in xrange(n):
		print "%4d : razlicnih: %4d      nenicelnih: %4d           min: %5.3f          max: %5.3f        avg: %5.3f        <0.5 : %4d      <0.5 : %4d " % \
				(i, razlicnih[i], nenicelnih[i], trainD.T[i].min(),trainD.T[i].max(),means[i],(trainD.T[i]<0.5).sum(),(trainD.T[i]>=0.5).sum())


