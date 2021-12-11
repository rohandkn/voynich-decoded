import numpy as np
import pdb

a = np.load("shap_sum.npy", allow_pickle=True)
div_list = []
for i in range(6):
	b = {}
	for k in a[i].keys():
		b[k] = a[i][k] / a[6][k]
	div_list.append({k: v for k, v in sorted(b.items(), key=lambda item: item[1])})

pdb.set_trace()