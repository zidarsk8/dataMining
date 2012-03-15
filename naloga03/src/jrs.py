import Orange
import cPickle
import random
import collections
import itertools
import os.path
from socket import gethostname
force_mini = True
data_dir = data_dir = "data/" if ("xgrid" in gethostname() and not force_mini)  else "minidata/"
orange_data_filename = data_dir + "trainingDataO.pickled"
orange_test_filename = data_dir + "testDataO.pickled"
orange_disc_filename = data_dir + "trainingDataOD.pickled"
orange_disc_test_filename = data_dir + "testDataOD.pickled"

class RawData:
    def __init__(self):
        self.labels_file_name = data_dir + "trainingLabels.csv"
        self.data_file_name = data_dir + "trainingData.csv"
        self.filtered_data_file_name = data_dir + "trainingDataT.csv"

        self.test_file_name = data_dir + "testData.csv"
        self.filtered_test_file_name = data_dir + "testDataT.csv"

        self.preprocess_labels()

    def remove_empty_features(self, threshold=10):
        """Remove features with fewer than threshold non-zero entries, save in CSV file."""
        f = open(self.data_file_name)
        self.m = len(f.readline().strip().split("\t"))
        print "Features:", self.m

        f.seek(0)
        dcount = collections.Counter() # non-empty data entries per feature
        for line in f:
            ind = [i for i, x in enumerate(line.strip().split("\t")) if x!="0"]
            dcount.update(ind)

        useful_features = set(k for k, v in dcount.items() if v >= threshold)
        self.m_red = len(useful_features)
        print "Columns with at least %d entries: %d (%4.2f%%)" % (threshold, self.m_red, 100*self.m_red/self.m)

        # column selection on training data set
        f.seek(0)
        selection = [x in useful_features for x in xrange(self.m)]
        fout = file(self.filtered_data_file_name, "w")
        for line in f:
            fout.write("\t".join(itertools.compress(line.strip().split("\t"), selection)) + "\n")
        fout.close()

        # column selection on test data set
        f = open(self.test_file_name)
        fout = file(self.filtered_test_file_name, "w")
        for line in f:
            fout.write("\t".join(itertools.compress(line.strip().split("\t"), selection)) + "\n")
        fout.close()

    def convert_to_orange(self):
        """Convert raw CVS data to Orange multi-target format, pickle the resulting data."""
        if not os.path.exists(self.filtered_data_file_name):
            print "Filtering empty features..."
            self.remove_empty_features()
        pass

        print "Reduced features:", self.m_red
        print "Labels:", len(self.labels)

        # set-up the domain
        class_vars = [Orange.feature.Discrete("c%s" % i, values=["F","T"]) for i in self.labels]
        features = [Orange.feature.Continuous("%d" % i) for i in range(self.m_red)]
        domain = Orange.data.Domain(features, False, class_vars=class_vars)

        # load the data from the csv files, construct Orange data table
        print "Loading from CVS and creating Orange data table..."
        data = Orange.data.Table(domain)

        ff = open(self.filtered_data_file_name)
        fl = open(self.labels_file_name)

        for fline, lline in itertools.izip(ff, fl):
            d = Orange.data.Instance(domain, [int(v) for v in fline.strip().split("\t")])
            dlabels = set(lline.strip().split(","))
            d.set_classes([["F", "T"][lab in dlabels] for lab in self.labels])
            data.append(d)
        print "Dumping training data..."
        cPickle.dump(data, file(orange_data_filename, "wb"), -1)

        # same with the test data set, class label set to None
        ff = open(self.filtered_test_file_name)

        # domain = Orange.data.Domain([Orange.feature.Continuous("%d" % i) for i in range(self.m_red)], False)
        domain = Orange.data.Domain(features, False)
        data = Orange.data.Table(domain)

        for fline in ff:
            d = Orange.data.Instance(domain, [int(v) for v in fline.strip().split("\t")])
            data.append(d)
        print "Dumping test data..."
        cPickle.dump(data, file(orange_test_filename, "wb"), -1)


    def preprocess_labels(self):
        """Store label profile (keys are labels, items labelled instances) in self.lprofile."""
        f = file(self.labels_file_name)
        self.lprofile = collections.defaultdict(set)
        self.n = 0
        for i, line in enumerate(f):
            self.n += 1
            labels = line.strip().split(",")
            for x in labels:
                self.lprofile[x].add(i)
        self.labels = self.lprofile.keys()

    def score_label_similarity(self, a, b):
        u = self.lprofile[a] & self.lprofile[b]
        return len(self.lprofile[a] & self.lprofile[b]) / float(len(self.lprofile[a] | self.lprofile[b])) if u else 0.

    def permute_lprofile(self):
        profile_items = range(self.n)
        for k, p in self.lprofile.items():
            self.lprofile[k] = set(random.sample(profile_items, len(p)))

class Data:
    def __init__(self, discretized=False, discretizer=Orange.feature.discretization.ThresholdDiscretizer(threshold=0.)):
        self.ml_data = cPickle.load(file(orange_disc_filename if discretized else orange_data_filename))
        self.test_data = cPickle.load(file(orange_disc_test_filename if discretized else orange_test_filename))
        print "Check train vs. test:", self.ml_data.domain.features[0] == self.test_data.domain.features[0]
        print "Loaded:", len(self.ml_data)
        self.classes = {v.name:v for v in self.ml_data.domain.class_vars}
        self.disc_features = [discretizer.constructVariable(x) for x in self.ml_data.domain.features]

    def get_single_class_data(self, label="c40"):
        """Construct a data set with given label as a class."""
        domain = Orange.data.Domain(self.ml_data.domain.features + [self.classes[label]])
        self.cdata = Orange.data.Table(domain, self.ml_data)
        return self.cdata

    def discretize(self, data):
        """Discretize features in the data set and dump to a file."""
        if data.domain.class_vars:
            domain = Orange.data.Domain(self.disc_features, False, class_vars = self.ml_data.domain.classVars)
        else:
            domain = Orange.data.Domain(self.disc_features, False)
        return Orange.data.Table(domain, data)

def add_dummy_class(test, train):
    domain = Orange.data.Domain(test.domain.features + [train.domain.class_var])
    return Orange.data.Table(domain, test)

def undefine(data):
    """All 0 entries in the data are set to None. Assumes attributes are continuous."""
    for d in data:
        for x in data.domain.features:
            if d[x] <= 0.:
                d[x] = None
