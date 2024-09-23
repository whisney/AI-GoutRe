import numpy as np
from sklearn.metrics import accuracy_score, roc_curve, f1_score, auc

def metric(test_label_, test_pred_):
    test_label = np.copy(test_label_)
    test_pred = np.copy(test_pred_)
    fpr, Sensitivitys, thresholds = roc_curve(test_label, test_pred)
    AUC = auc(fpr, Sensitivitys)
    Specifcitys = 1 - fpr
    index = np.argmin(np.abs(Sensitivitys - Specifcitys))
    Sensitivity = Sensitivitys[index]
    Specifcity = 1 - fpr[index]
    threshold = thresholds[index]
    test_pred[test_pred <= threshold] = 0
    test_pred[test_pred > threshold] = 1
    ACC = accuracy_score(test_label, test_pred)
    F1 = f1_score(test_label, test_pred)
    return AUC, ACC, Specifcity, Sensitivity, F1


def calculate_sensitivity_specificity(actual, predicted):
    # 计算真阳性、假阴性、真阴性、假阳性
    TP = sum((actual == 1) & (predicted == 1))
    FN = sum((actual == 1) & (predicted == 0))
    TN = sum((actual == 0) & (predicted == 0))
    FP = sum((actual == 0) & (predicted == 1))
    # 计算敏感度和特异度
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    return sensitivity, specificity

def metric_get_threshold(test_label_, test_pred_, threshold_=None):
    test_label = np.copy(test_label_)
    test_pred = np.copy(test_pred_)
    fpr, Sensitivitys, thresholds = roc_curve(test_label, test_pred)
    AUC = auc(fpr, Sensitivitys)
    Specifcitys = 1 - fpr
    if threshold_ == None:
        index = np.argmin(np.abs(Sensitivitys - Specifcitys))
        Sensitivity = Sensitivitys[index]
        Specifcity = 1 - fpr[index]
        threshold = thresholds[index]
        test_pred[test_pred <= threshold] = 0
        test_pred[test_pred > threshold] = 1
    else:
        threshold = threshold_
        test_pred[test_pred <= threshold] = 0
        test_pred[test_pred > threshold] = 1
        Sensitivity, Specifcity = calculate_sensitivity_specificity(test_label, test_pred)
    ACC = accuracy_score(test_label, test_pred)
    F1 = f1_score(test_label, test_pred)
    return AUC, ACC, Specifcity, Sensitivity, F1, threshold