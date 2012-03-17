from time import time
import cPickle
import sys
import Orange
import jrs
reload(jrs)

#def prepairCsv(filter=10):
#	#change empty features to 10 for real results and 40+ for testing
#	raw = jrs.RawData()
#	raw.remove_empty_features(filter)
#	raw.convert_to_orange()
#
#prepairCsv(50)

numOfTrees = 5

mld = jrs.Data()

trainSet = mld.ml_data
testSet = mld.test_data

forest = Orange.ensemble.forest.RandomForestLearner(trees = numOfTrees)

tl = len(testSet)
cl = len(mld.classes)

results = [{} for x in xrange(tl)]
print "starting random forest"
for ci, clas in enumerate(mld.classes):
	model = forest(mld.get_single_class_data(clas))
	for index, row in enumerate(testSet):
		#if index % 100 == 0:
		sys.stdout.write("\rcalculating: %3d%%" % (((ci*tl)+index)*100/(tl*cl)))
		sys.stdout.flush()
		res = model(row,Orange.classification.Classifier.GetProbabilities)["T"]
		if res != 0:
			results[index][clas] = res
print

sortedRes = [sorted(x.iteritems(), key=lambda x: x[1], reverse=True) for x in results]
		
for i in sortedRes[1]:
	print i 

cPickle.dump(sortedRes,open("minidata/rf-sotedRes-%d-trees-%d.pickle" % (numOfTrees,time()),"w"))


