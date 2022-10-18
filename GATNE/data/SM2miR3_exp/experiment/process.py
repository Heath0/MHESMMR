"""
@File    : process.py
@Time    : 2022/4/14 11:43
"""
import pandas as pd
import numpy as np
from tqdm import trange

if __name__ == '__main__':
    gatne_feature = pd.read_csv(r'type1.csv', header=None).values.tolist()
    attri_feature = pd.read_csv(r'../feature/attri_line_128.txt',header=None,sep=' ').values.tolist()

    name = []
    feature = []

    for i in range(len(gatne_feature)):
        name.append(gatne_feature[i][0])
    print(name)

    for i in range(len(attri_feature)):
        if attri_feature[i][0] in name:
            for a in gatne_feature:
                if a[0] == attri_feature[i][0]:
                    feature.append(a)
        else:
            feature.append(attri_feature[i])
    pd.DataFrame(feature).to_csv(r'feature_GATNE_non-attri_line_up-regulate_128.csv',index=None,header=None)


    gatne_feature = pd.read_csv(r'type2.csv', header=None).values.tolist()

    name = []
    feature = []

    for i in range(len(gatne_feature)):
        name.append(gatne_feature[i][0])
    print(name)

    for i in range(len(attri_feature)):
        if attri_feature[i][0] in name:
            for a in gatne_feature:
                if a[0] == attri_feature[i][0]:
                    feature.append(a)
        else:
            feature.append(attri_feature[i])
    pd.DataFrame(feature).to_csv(r'feature_GATNE_non-attri_line_down-regulate_128.csv',index=None,header=None)