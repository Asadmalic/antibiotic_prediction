
import numpy as np

import os

fname = os.path.join(
    os.getcwd(),
    "features_gpa_expr_snps",
    'features_gpa_expr_snps',
    "genexp",
    "genexp_feature_vect.npz"
)
d = np.load(fname)

for k in d.keys():
    print(k, d[k].shape)

fname = os.path.join(
    os.getcwd(),
    "features_gpa_expr_snps",
    'features_gpa_expr_snps',
    "gpa",
    "gpa_feature_vect.npz"
)
d = np.load(fname)

for k in d.keys():
    print(k, d[k].shape)

fname = os.path.join(
    os.getcwd(),
    "features_gpa_expr_snps",
    'features_gpa_expr_snps',
    "snps",
    "snps_feature_vect.npz"
)
d = np.load(fname)

for k in d.keys():
    print(k, d[k].shape)


