import sys
import Orange
import jrs
reload(jrs)

#raw = jrs.RawData()
#raw.remove_empty_features(40)
#raw.convert_to_orange()
#exit()

mld = jrs.Data()

trainSet = mld.ml_data
testSet = mld.test_data

forest = Orange.ensemble.forest.RandomForestLearner(trees = 2)

tl = len(testSet)
cl = len(mld.classes)

results = [{} for x in xrange(tl)]
for ci, clas in enumerate(mld.classes):
	model = forest(mld.get_single_class_data(clas))
	for index, row in enumerate(testSet):
		if index % 100 == 0:
			sys.stdout.write("\rcalculating: %3d%%" % (((ci*tl)+index)*100/(tl*cl)))
			sys.stdout.flush()
		res = model(row,Orange.classification.Classifier.GetProbabilities)["T"]
		if res != 0:
			results[index][clas] = res
		
print


def sk(x): return x[1]
dd = sorted(results[1].iteritems(), key=sk, reverse=True)
		
print dd
