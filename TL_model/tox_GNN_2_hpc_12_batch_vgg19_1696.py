import os
import pandas as pd


path="/rds/user/dh684/hpc-work/kekulescope/transfer/endpoints_tox21_7831_vgg19_1696.csv"#input file csv with smiles label
f=pd.read_csv(path)
for i in f.columns[0:-1]:
    os.system("MKL_THREADING_LAYER=GPU python tox_GNN_2_hpc_12_2_stra_vgg19_1696.py " +i )