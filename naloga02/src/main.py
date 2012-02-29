import Orange
import random
import numpy as np
import jrs
reload(jrs)

alfa = 0.05
shuffleCount = 500
mld=jrs.Data(discretized=True)

gain = Orange.feature.scoring.InfoGain()

#uporabniAttributi = {}
for cn in mld.classes.keys():
	data = mld.get_single_class_data(label=cn)
	ig = np.array([gain(feature, data) for feature in data.domain.features])
	suma = np.zeros(len(data[0])-1)

	for s in xrange(shuffleCount):
		c = [d.get_class() for d in data]
		random.shuffle(c)
		[d.set_class(c[i]) for i,d in enumerate(data)]

		igp = np.array([gain(feature, data) for feature in data.domain.features])
		suma += np.greater(igp, ig)

	suma /= shuffleCount
	t = np.nonzero(suma<alfa)
	np.save(cn, t)
	print "%s done %d" % (cn, len(t[0]))
	#uporabniAttributi[cn] = np.nonzero(suma<alfa)