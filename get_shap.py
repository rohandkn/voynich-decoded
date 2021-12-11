import numpy as np
import pdb

a = np.load("shap_sum.npy", allow_pickle=True)
c = []
for i in range(6):
	b = {}
	for k in a[0].keys():
		b[k] = a[0][k] / a[6][k]
	c.append({k: v for k, v in sorted(b.items(), key=lambda item: item[1])})

pdb.set_trace()