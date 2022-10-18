import pandas as pd
import numpy as np
import warnings

from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost.sklearn import XGBClassifier
import lightgbm as lgb
import math

def MyLabel(Sample):
    label = []
    for i in range(int(len(Sample) / 2)):
        label.append(1)
    for i in range(int(len(Sample) / 2)):
        label.append(0)
    return label

def GenerateBehaviorFeature(InteractionPair, NodeBehavior):
    SampleFeature1 = []
    SampleFeature2 = []
    for i in range(len(InteractionPair)):
        Pair1 = str(InteractionPair[i][0])#miRNA
        Pair2 = str(InteractionPair[i][1])#drug

        for m in range(len(NodeBehavior)):
            if Pair1 == NodeBehavior[m][0]:
                SampleFeature1.append(NodeBehavior[m][1:])
                break

        for n in range(len(NodeBehavior)):
            if Pair2 == NodeBehavior[n][0]:
                SampleFeature2.append(NodeBehavior[n][1:])
                break

    SampleFeature1 = np.array(SampleFeature1).astype('float32')
    SampleFeature2 = np.array(SampleFeature2).astype('float32')

    return SampleFeature1, SampleFeature2

def GenerateAttributeFeature(InteractionPair, drug_feature, miRNA_feature):
    SampleFeature1 = []
    SampleFeature2 = []
    for i in range(len(InteractionPair)):
        Pair1 = str(InteractionPair[i][1])  #drug
        Pair2 = str(InteractionPair[i][0])  #mirna
        for m in range(len(drug_feature)):#drug
            if int(Pair1) == int(drug_feature[m][0]):
                SampleFeature1.append(drug_feature[m][1:])
                break
        for n in range(len(miRNA_feature)):#mirna
            if Pair2 == str(miRNA_feature[n][0]):
                SampleFeature2.append(miRNA_feature[n][1:])
                break

    SampleFeature1 = np.array(SampleFeature1).astype('float32')
    SampleFeature2 = np.array(SampleFeature2).astype('float32')

    return SampleFeature1, SampleFeature2

#生成混淆矩阵
def MyConfusionMatrix(y_real,y_predict):
    from sklearn.metrics import confusion_matrix
    CM = confusion_matrix(y_real, y_predict)
    print(CM)
    CM = CM.tolist()
    TN = CM[0][0]
    FP = CM[0][1]
    FN = CM[1][0]
    TP = CM[1][1]
    print('TN:%d, FP:%d, FN:%d, TP:%d' % (TN, FP, FN, TP))
    Acc = (TN + TP) / (TN + TP + FN + FP)
    Sen = TP / (TP + FN)
    Spec = TN / (TN + FP)
    Prec = TP / (TP + FP)
    MCC = (TP * TN - FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    # 分母可能出现0，需要讨论待续
    print('Acc:', round(Acc, 4))
    print('Sen:', round(Sen, 4))
    print('Spec:', round(Spec, 4))
    print('Prec:', round(Prec, 4))
    print('MCC:', round(MCC, 4))
    Result = []
    Result.append(round(Acc, 4))
    Result.append(round(Sen, 4))
    Result.append(round(Spec, 4))
    Result.append(round(Prec, 4))
    Result.append(round(MCC, 4))
    return Result

#
def MyAverage(matrix):
    SumAcc = 0
    SumSen = 0
    SumSpec = 0
    SumPrec = 0
    SumMcc = 0
    counter = 0
    while counter < len(matrix):
        SumAcc = SumAcc + matrix[counter][0]
        SumSen = SumSen + matrix[counter][1]
        SumSpec = SumSpec + matrix[counter][2]
        SumPrec = SumPrec + matrix[counter][3]
        SumMcc = SumMcc + matrix[counter][4]
        counter = counter + 1
    print('AverageAcc:',SumAcc / len(matrix))
    print('AverageSen:', SumSen / len(matrix))
    print('AverageSpec:', SumSpec / len(matrix))
    print('AveragePrec:', SumPrec / len(matrix))
    print('AverageMcc:', SumMcc / len(matrix))
    return

#标准差
def MyStd(result):
    import numpy as np
    NewMatrix = []
    counter = 0
    while counter < len(result[0]):
        row = []
        NewMatrix.append(row)
        counter = counter + 1
    counter = 0
    while counter < len(result):
        counter1 = 0
        while counter1 < len(result[counter]):
            NewMatrix[counter1].append(result[counter][counter1])
            counter1 = counter1 + 1
        counter = counter + 1
    StdList = []
    MeanList = []
    counter = 0
    while counter < len(NewMatrix):
        # std
        arr_std = np.std(NewMatrix[counter], ddof=1)
        StdList.append(arr_std)
        # mean
        arr_mean = np.mean(NewMatrix[counter])
        MeanList.append(arr_mean)
        counter = counter + 1


    result.append(MeanList)
    result.append(StdList)
    # 换算成百分比制
    counter = 0
    while counter < len(result):
        counter1 = 0
        while counter1 < len(result[counter]):
            result[counter][counter1] = round(result[counter][counter1] * 100, 2)
            counter1 = counter1 + 1
        counter = counter + 1

    counter = 0
    while counter < len(StdList):
        result[5][counter] = str(result[5][counter]) + str('+') + str(result[6][counter])
        counter = counter + 1

    return result

######生成特征向量#############
featuer = pd.read_csv(r'../../../Data/SM2miR_exp/feature/feature_GATNE_line_up-regulate_128.csv', sep=',', header=None).values.tolist()
AllConfusionMatrix = [] #每一折的混淆矩阵

for i in range(1,6,1):
    # 划分为5折交叉验证数据集
    # X_train, X_test, Y_train_labels, Y_test_labels = train_test_split(x_features, y_labels, test_size=0.2,stratify=y_labels,random_state=2021+i)  #按y比例分层抽样，通过用于分类问题    print(train_X)
    train_set = pd.read_csv(r'../../../Data/SM2miR_exp/5-fold/Fold'+str(i)+' upregulate train.csv',header=0)
    test_set = pd.read_csv(r'../../../Data/SM2miR_exp/5-fold/Fold'+str(i)+' upregulate test.csv',header=0)
    print(train_set.columns)

    train_labels = train_set['label']
    test_labels =test_set['label']
    print('train_labels shape',np.array(train_labels).shape,)
    print('test_labels shape',np.array(test_labels).shape)

    train_x1_behavior,train_x2_behavior = GenerateBehaviorFeature(train_set.values.tolist(),featuer)
    test_x1_behavior, test_x2_behavior = GenerateBehaviorFeature(test_set.values.tolist(), featuer)
    print('train_x1_behavior shape', train_x1_behavior.shape)
    print('train_x2_behavior shape', train_x2_behavior.shape)

    train_features = np.hstack((train_x1_behavior,train_x2_behavior))
    test_feature = np.hstack((test_x1_behavior,test_x2_behavior))
    print('x_features shape',train_features.shape,test_feature.shape)

    lgb_cf = lgb.LGBMClassifier(random_state=2021+i)  #lgb

    lgb_cf.fit(train_features, train_labels)

    prediction_score = lgb_cf.score(test_feature, test_labels)
    print(prediction_score)

    y_test_Pre = lgb_cf.predict(test_feature)
    y_test_Prob = lgb_cf.predict_proba(test_feature)

    y_test_result_pre = []
    y_test_result_prob = []
    for counter in range(len(y_test_Pre)):
        row = []
        row.append(test_labels[counter])
        row.append(y_test_Pre[counter])
        y_test_result_pre.append(row)
        row_prob = []
        row_prob.append(test_labels[counter])
        row_prob.append(y_test_Prob[counter][1])
        y_test_result_prob.append(row_prob)

    pd.DataFrame(y_test_result_pre).to_csv(r'./y_test_Pre'+str(i)+'.csv', header=None, index=None)
    pd.DataFrame(y_test_result_prob).to_csv(r'./y_test_Prob'+str(i)+'.csv', header=None, index=None)

    ConfusionMatrix = MyConfusionMatrix(test_labels,y_test_Pre) #单折混淆矩阵
    print('AUC:', prediction_score)
    ConfusionMatrix.append(AllConfusionMatrix)