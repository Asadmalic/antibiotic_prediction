
import os
import numpy as np
import pandas as pd

data_np = np.load(os.path.join(os.getcwd(), "features_gpa_expr_snps", "genexp", "genexp_feature_vect.npz"), mmap_mode='r')
data_rs = data_np['data'].reshape(416,6026)


a_file = open(os.path.join(os.getcwd(), "features_gpa_expr_snps", "genexp", "genexp_feature_list.txt"), "r")
column_names = [(line.strip()).split() for line in a_file]
a_file.close()

columns = [val[0] if len(val)==1 else str(val) for val in column_names]

data = pd.DataFrame(data_rs, columns=columns)

target_file = pd.read_csv(os.path.join(os.getcwd(), "metadata", "phenotypes.txt"), sep='\t')

def data_prep(target_name):
    assert target_name in ['Meropenem_S-vs-R', 'Ciprofloxacin_S-vs-R', 'Ceftazidim_S-vs-R']
    data[target_name.split('_')[0]] = target_file[target_name]
    return data

