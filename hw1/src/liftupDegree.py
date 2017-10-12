import os
import sys
import numpy as np

in_csv = sys.argv[1]
degree = int(sys.argv[2])

name, ext = os.path.splitext(in_csv)
out_csv = name + "_degree" + str(degree) + ext

with open(out_csv, "w") as outf:
	with open(in_csv, "r") as inf:
		header = inf.readline()
		outf.write(header)

		for line in inf:
			vals = line.strip().split(' ')
			idx, feats, label = vals[0], vals[1:-1], vals[-1]

			first_deg = np.asarray(feats, dtype=np.float32)

			final_feat = first_deg
			for d in range(2, degree+1):
				final_feat = np.append(final_feat, first_deg ** d)
				#final_feat = np.reshape(final_feat, (-1, 1)) * first_deg
				#final_feat = np.reshape(final_feat, -1) 

			outf.write(idx + " ")
			for feat in final_feat:
				outf.write(str(feat) + " ")
			outf.write(label)
			outf.write("\n")
