import jrs

raw = jrs.RawData()
raw.remove_empty_features()
raw.convert_to_orange()

data = jrs.Data()
ddata = data.discretize()