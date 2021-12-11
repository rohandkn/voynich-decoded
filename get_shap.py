import numpy as np
import pdb

a = np.load("shap_sum.npy", allow_pickle=True)
b = {}
for k in a[0].keys():
	b[k] = a[0][k] / a[6][k]
b0 = {k: v for k, v in sorted(b.items(), key=lambda item: item[1])}
pdb.set_trace()