import Orange
import cPickle


data = Orange.data.Table("data/train.tab")

reliefScore = {}
for attr in data.domain.attributes:
    reliefScore[attr] = Orange.feature.scoring.Relief(attr, data)
    print attr,reliefScore[attr]
    
cPickle.dump(reliefScore,open("reliefScore.pkl","w"))