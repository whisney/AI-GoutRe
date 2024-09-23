import os
import random
from utils import metric
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer, KNNImputer
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import PowerTransformer, MinMaxScaler, StandardScaler
import joblib
from scipy.stats import pearsonr
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from sklearn.linear_model import LogisticRegressionCV, LassoCV
from skfeature.function.information_theoretical_based import CIFE
from skfeature.function.information_theoretical_based import CMIM
from skfeature.function.information_theoretical_based import DISR
from skfeature.function.information_theoretical_based import ICAP
from skfeature.function.information_theoretical_based import JMI
from skfeature.function.information_theoretical_based import MIFS
from skfeature.function.information_theoretical_based import MIM
from skfeature.function.information_theoretical_based import MRMR
from skfeature.function.similarity_based import fisher_score
from skfeature.function.similarity_based import lap_score
from skfeature.function.similarity_based import reliefF
from skfeature.function.similarity_based import SPEC
from skfeature.function.similarity_based import trace_ratio
from skfeature.function.sparse_learning_based import ll_l21
from skfeature.function.sparse_learning_based import ls_l21
from skfeature.function.sparse_learning_based import MCFS
from skfeature.function.sparse_learning_based import RFS
from skfeature.function.sparse_learning_based import UDFS
from skfeature.function.statistical_based import CFS
from skfeature.function.statistical_based import f_score
from skfeature.function.statistical_based import gini_index
from skfeature.function.statistical_based import t_score
from skfeature.function.streaming import alpha_investing
from skfeature.utility import construct_W
from skfeature.utility.sparse_learning import feature_ranking, construct_label_matrix_pan
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.naive_bayes import GaussianNB
from utils_ANN import train_ANN
import torch

seed = 0
selected_feature_num = 20
use_feature_mode = 'all'
# use_feature_mode = 'objective'
train_center_id = ['0', '2', '3', '4']
test_center_id = ['1']
prospective_center_id = ['5']

save_dir = 'trained_models/center{}_center{}_center{}'.format(''.join(train_center_id), ''.join(test_center_id), ''.join(prospective_center_id))

original_df = []
external1_df = []
prospective_df = []
for id in train_center_id:
    original_df.append(pd.read_excel(r'data/Center{}.xlsx'.format(id)))
for id in test_center_id:
    external1_df.append(pd.read_excel(r'data/Center{}.xlsx'.format(id)))
for id in prospective_center_id:
    prospective_df.append(pd.read_excel(r'data/Center{}.xlsx'.format(id)))
original_df = pd.concat(original_df, axis=0, ignore_index=True)
external1_df = pd.concat(external1_df, axis=0, ignore_index=True)
prospective_df = pd.concat(prospective_df, axis=0, ignore_index=True)

original_df.drop(columns=['AdmissionTime', 'DischargeTime', 'flaredate', 'Primary admission diagnosis'], inplace=True)
external1_df.drop(columns=['AdmissionTime', 'DischargeTime', 'flaredate', 'Primary admission diagnosis'], inplace=True)
prospective_df.drop(columns=['AdmissionTime', 'DischargeTime', 'flaredate', 'Primary admission diagnosis'], inplace=True)

if use_feature_mode == 'objective':
    drop_feature_list = ['Inpatient Date', 'Allopurinol', 'Febuxostat', 'Benzbromarone', 'UAE', 'Fenofibrate',
                           'SGLT2 Inhibitors', 'Statins', 'CCB', 'Losartan', 'NL-ARBs', 'Thiazide', 'Loop', 'L&T',
                           'βblocker', 'Low-dose ASA', 'Mannitol', 'NaHCO₃', 'GCs']
    original_df.drop(columns=drop_feature_list, inplace=True)
    external1_df.drop(columns=drop_feature_list, inplace=True)
    prospective_df.drop(columns=drop_feature_list, inplace=True)

ID_list = list(original_df['unique_id'])
label_list = list(original_df['Group'])

split_dir = os.path.join(save_dir, 'use_{}_features'.format(use_feature_mode), 'train_val_seed{}'.format(seed))
train_val_sfolder = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
train_ID_list = []
val_ID_list = []
for train_index, val_index in train_val_sfolder.split(ID_list, label_list):
    for id in train_index:
        train_ID_list.append(ID_list[id])
    for id in val_index:
        val_ID_list.append(ID_list[id])
os.makedirs(split_dir, exist_ok=True)
df_train_ID = pd.DataFrame({'ID': train_ID_list})
df_train_ID.to_excel(os.path.join(split_dir, 'train_ID.xlsx'), index=False)
df_val_ID = pd.DataFrame({'ID': val_ID_list})
df_val_ID.to_excel(os.path.join(split_dir, 'val_ID.xlsx'), index=False)

train_ID_list = list(pd.read_excel(os.path.join(split_dir, 'train_ID.xlsx'))['ID'])
val_ID_list = list(pd.read_excel(os.path.join(split_dir, 'val_ID.xlsx'))['ID'])

df_train_original = original_df[original_df.unique_id.isin(train_ID_list)]
df_val_original = original_df[original_df.unique_id.isin(val_ID_list)]
df_external1_original = external1_df.copy()
df_prospective_original = prospective_df.copy()

for Imputer_name in ['IterativeImputer', 'MeanImputer', 'MedianImputer', 'KNNImputer']:
    print('Imputer_name: {}'.format(Imputer_name))
    Imputer_dir = os.path.join(split_dir, Imputer_name)
    os.makedirs(Imputer_dir, exist_ok=True)
    if os.path.exists(os.path.join(Imputer_dir, 'train_data_Imputered.xlsx')) and \
            os.path.exists(os.path.join(Imputer_dir, 'val_data_Imputered.xlsx')):
        df_train_Imputered = pd.read_excel(os.path.join(Imputer_dir, 'train_data_Imputered.xlsx'))
        df_val_Imputered = pd.read_excel(os.path.join(Imputer_dir, 'val_data_Imputered.xlsx'))
        df_external1_Imputered = pd.read_excel(os.path.join(Imputer_dir, 'external1_data_Imputered.xlsx'))
        df_prospective_Imputered = pd.read_excel(os.path.join(Imputer_dir, 'prospective_data_Imputered.xlsx'))
    else:
        df_train_Imputered = df_train_original.copy()
        df_val_Imputered = df_val_original.copy()
        df_external1_Imputered = df_external1_original.copy()
        df_prospective_Imputered = df_prospective_original.copy()
        train_features = df_train_Imputered.iloc[:, 2:].values
        val_features = df_val_Imputered.iloc[:, 2:].values
        external1_features = df_external1_Imputered.iloc[:, 2:].values
        prospective_features = df_prospective_Imputered.iloc[:, 2:].values
        if Imputer_name == 'IterativeImputer':
            imp = IterativeImputer(max_iter=10, random_state=seed)
        elif Imputer_name == 'MeanImputer':
            imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        elif Imputer_name == 'MeanImputer':
            imp = SimpleImputer(missing_values=np.nan, strategy='median')
        elif Imputer_name == 'KNNImputer':
            imp = KNNImputer(n_neighbors=5)
        imp.fit(train_features)
        train_features = imp.transform(train_features)
        val_features = imp.transform(val_features)
        external1_features = imp.transform(external1_features)
        prospective_features = imp.transform(prospective_features)
        df_train_Imputered.iloc[:, 2:] = train_features
        df_val_Imputered.iloc[:, 2:] = val_features
        df_external1_Imputered.iloc[:, 2:] = external1_features
        df_prospective_Imputered.iloc[:, 2:] = prospective_features
        joblib.dump(imp, os.path.join(Imputer_dir, 'Imputer.pkl'))
        df_train_Imputered.to_excel(os.path.join(Imputer_dir, 'train_data_Imputered.xlsx'), index=False)
        df_val_Imputered.to_excel(os.path.join(Imputer_dir, 'val_data_Imputered.xlsx'), index=False)
        df_external1_Imputered.to_excel(os.path.join(Imputer_dir, 'external1_data_Imputered.xlsx'), index=False)
        df_prospective_Imputered.to_excel(os.path.join(Imputer_dir, 'prospective_data_Imputered.xlsx'), index=False)

    for Normalizer_name in ['Yeo_Johnson', 'MinMax', 'Standard']:
        print('Normalizer_name: {}'.format(Normalizer_name))
        Normalizer_dir = os.path.join(Imputer_dir, Normalizer_name)
        os.makedirs(Normalizer_dir, exist_ok=True)
        if os.path.exists(os.path.join(Normalizer_dir, 'train_data_Normalizered.xlsx')) and \
                os.path.exists(os.path.join(Normalizer_dir, 'val_data_Normalizered.xlsx')) and \
                os.path.exists(os.path.join(Normalizer_dir, 'remove_features.xlsx')):
            df_train_Normalizered = pd.read_excel(os.path.join(Normalizer_dir, 'train_data_Normalizered.xlsx'))
            df_val_Normalizered = pd.read_excel(os.path.join(Normalizer_dir, 'val_data_Normalizered.xlsx'))
            df_external1_Normalizered = pd.read_excel(os.path.join(Normalizer_dir, 'external1_data_Normalizered.xlsx'))
            df_prospective_Normalizered = pd.read_excel(os.path.join(Normalizer_dir, 'prospective_data_Normalizered.xlsx'))
            remove_features_name = list(pd.read_excel(os.path.join(Normalizer_dir, 'remove_features.xlsx'))['Feature'])
            df_train_Normalizered.drop(columns=remove_features_name, inplace=True)
            df_val_Normalizered.drop(columns=remove_features_name, inplace=True)
            df_external1_Normalizered.drop(columns=remove_features_name, inplace=True)
            df_prospective_Normalizered.drop(columns=remove_features_name, inplace=True)
        else:
            df_train_Normalizered = df_train_Imputered.copy()
            df_val_Normalizered = df_val_Imputered.copy()
            df_external1_Normalizered = df_external1_Imputered.copy()
            df_prospective_Normalizered = df_prospective_Imputered.copy()
            train_features = df_train_Imputered.iloc[:, 2:].values
            val_features = df_val_Imputered.iloc[:, 2:].values
            external1_features = df_external1_Imputered.iloc[:, 2:].values
            prospective_features = df_prospective_Imputered.iloc[:, 2:].values
            if Normalizer_name == 'Yeo_Johnson':
                Scaler = PowerTransformer(method='yeo-johnson')
            elif Normalizer_name == 'MinMax':
                Scaler = MinMaxScaler(feature_range=(0, 1))
            elif Normalizer_name == 'Standard':
                Scaler = StandardScaler()
            Scaler.fit(train_features)
            train_features = Scaler.transform(train_features)
            val_features = Scaler.transform(val_features)
            external1_features = Scaler.transform(external1_features)
            prospective_features = Scaler.transform(prospective_features)
            df_train_Normalizered.iloc[:, 2:] = train_features
            df_val_Normalizered.iloc[:, 2:] = val_features
            df_external1_Normalizered.iloc[:, 2:] = external1_features
            df_prospective_Normalizered.iloc[:, 2:] = prospective_features
            joblib.dump(Scaler, os.path.join(Normalizer_dir, 'Normalizer.pkl'))
            df_train_Normalizered.to_excel(os.path.join(Normalizer_dir, 'train_data_Normalizered.xlsx'), index=False)
            df_val_Normalizered.to_excel(os.path.join(Normalizer_dir, 'val_data_Normalizered.xlsx'), index=False)
            df_external1_Normalizered.to_excel(os.path.join(Normalizer_dir, 'external1_data_Normalizered.xlsx'), index=False)
            df_prospective_Normalizered.to_excel(os.path.join(Normalizer_dir, 'prospective_data_Normalizered.xlsx'),index=False)

            feature_name = list(df_train_Normalizered.columns[2:])
            feature_name_copy = feature_name.copy()
            choosed_features_name = []
            remove_features_name = []
            for name in feature_name:
                choose = 1
                feature_name_copy.remove(name)
                for compared_name in feature_name_copy:
                    P = pearsonr(df_train_Normalizered[name], df_train_Normalizered[compared_name])
                    if abs(P[0]) > 0.8:
                        choose = 0
                        remove_features_name.append(name)
                        break
                if choose:
                    choosed_features_name.append(name)

            df_train_features = df_train_Normalizered[choosed_features_name]
            df_train_features_ac = add_constant(df_train_features)
            Variable_name_list = list(df_train_features_ac.columns[1:])
            vif_list = [variance_inflation_factor(df_train_features_ac.values, i) for i in range(df_train_features_ac.shape[1])][1:]
            for i, Variable_name in enumerate(Variable_name_list):
                if vif_list[i] > 10:
                    choosed_features_name.remove(Variable_name)
                    remove_features_name.append(Variable_name)

            df_choosed_features = pd.DataFrame({'Feature': choosed_features_name})
            df_remove_features = pd.DataFrame({'Feature': remove_features_name})
            df_choosed_features.to_excel(os.path.join(Normalizer_dir, 'choosed_features.xlsx'), index=False)
            df_remove_features.to_excel(os.path.join(Normalizer_dir, 'remove_features.xlsx'), index=False)

            df_train_Normalizered.drop(columns=remove_features_name, inplace=True)
            df_val_Normalizered.drop(columns=remove_features_name, inplace=True)
            df_external1_Normalizered.drop(columns=remove_features_name, inplace=True)
            df_prospective_Normalizered.drop(columns=remove_features_name, inplace=True)

        for Selector_name in ['CIFE', 'CMIM', 'DISR', 'ICAP', 'JMI', 'MIFS', 'MIM', 'MRMR', 'fisher_score',
                              'lap_score', 'reliefF', 'SPEC', 'trace_ratio', 'll_l21', 'ls_l21', 'MCFS',
                              'RFS', 'UDFS', 'CFS', 'f_score', 'gini_index', 't_score', 'alpha_investing',
                              'LASSO']:
            print('Selector_name: {}'.format(Selector_name))
            Selector_dir = os.path.join(Normalizer_dir, Selector_name)
            os.makedirs(Selector_dir, exist_ok=True)
            if os.path.exists(os.path.join(Selector_dir, 'train_data_Selectored.xlsx')) and \
                    os.path.exists(os.path.join(Selector_dir, 'val_data_Selectored.xlsx')):
                df_train_Selectored = pd.read_excel(os.path.join(Selector_dir, 'train_data_Selectored.xlsx'))
                df_val_Selectored = pd.read_excel(os.path.join(Selector_dir, 'val_data_Selectored.xlsx'))
                df_external1_Selectored = pd.read_excel(os.path.join(Selector_dir, 'external1_data_Selectored.xlsx'))
                df_prospective_Selectored = pd.read_excel(os.path.join(Selector_dir, 'prospective_data_Selectored.xlsx'))
            else:
                feature_name_list = list(df_train_Normalizered.columns[2:])
                X_train = np.array(df_train_Normalizered.iloc[:, 2:])
                y_train = np.array(df_train_Normalizered.iloc[:, 1])
                df_train_Selectored = df_train_Normalizered.copy()
                df_val_Selectored = df_val_Normalizered.copy()
                df_external1_Selectored = df_external1_Normalizered.copy()
                df_prospective_Selectored = df_prospective_Normalizered.copy()
                if Selector_name == 'CIFE':
                    idx, _, _ = CIFE.cife(X_train, y_train, n_selected_features=selected_feature_num)
                elif Selector_name == 'CMIM':
                    idx, _, _ = CMIM.cmim(X_train, y_train, n_selected_features=selected_feature_num)
                elif Selector_name == 'DISR':
                    idx, _, _ = DISR.disr(X_train, y_train, n_selected_features=selected_feature_num)
                elif Selector_name == 'ICAP':
                    idx, _, _ = ICAP.icap(X_train, y_train, n_selected_features=selected_feature_num)
                elif Selector_name == 'JMI':
                    idx, _, _ = JMI.jmi(X_train, y_train, n_selected_features=selected_feature_num)
                elif Selector_name == 'MIFS':
                    idx, _, _ = MIFS.mifs(X_train, y_train, n_selected_features=selected_feature_num)
                elif Selector_name == 'MIM':
                    idx, _, _ = MIM.mim(X_train, y_train, n_selected_features=selected_feature_num)
                elif Selector_name == 'MRMR':
                    idx, _, _ = MRMR.mrmr(X_train, y_train, n_selected_features=selected_feature_num)
                elif Selector_name == 'fisher_score':
                    score = fisher_score.fisher_score(X_train, y_train)
                    idx = fisher_score.feature_ranking(score)[:selected_feature_num]
                elif Selector_name == 'lap_score':
                    kwargs_W = {"metric": "euclidean", "neighbor_mode": "knn", "weight_mode": "heat_kernel", "k": 5, 't': 1}
                    W = construct_W.construct_W(X_train, **kwargs_W)
                    score = lap_score.lap_score(X_train, W=W)
                    idx = lap_score.feature_ranking(score)[:selected_feature_num]
                elif Selector_name == 'reliefF':
                    score = reliefF.reliefF(X_train, y_train)
                    idx = reliefF.feature_ranking(score)[:selected_feature_num]
                elif Selector_name == 'SPEC':
                    kwargs_style = {'style': 0}
                    score = SPEC.spec(X_train, **kwargs_style)
                    idx = SPEC.feature_ranking(score, **kwargs_style)[:selected_feature_num]
                elif Selector_name == 'trace_ratio':
                    idx, _, _ = trace_ratio.trace_ratio(X_train, y_train, selected_feature_num, style='fisher')
                elif Selector_name == 'll_l21':
                    Y_train = construct_label_matrix_pan(y_train)
                    Weight, obj, value_gamma = ll_l21.proximal_gradient_descent(X_train, Y_train, 0.1, verbose=False)
                    idx = feature_ranking(Weight)[:selected_feature_num]
                elif Selector_name == 'ls_l21':
                    Y_train = construct_label_matrix_pan(y_train)
                    Weight, obj, value_gamma = ls_l21.proximal_gradient_descent(X_train, Y_train, 0.1, verbose=False)
                    idx = feature_ranking(Weight)[:selected_feature_num]
                elif Selector_name == 'MCFS':
                    kwargs_W = {"metric": "euclidean", "neighborMode": "knn", "weightMode": "heatKernel", "k": 5, 't': 1}
                    W = construct_W.construct_W(X_train, **kwargs_W)
                    num_cluster = 2
                    Weight = MCFS.mcfs(X_train, n_selected_features=selected_feature_num, W=W, n_clusters=num_cluster)
                    idx = MCFS.feature_ranking(Weight)[:selected_feature_num]
                elif Selector_name == 'RFS':
                    Y_train = construct_label_matrix_pan(y_train)
                    Weight = RFS.rfs(X_train, Y_train, gamma=0.1)
                    idx = feature_ranking(Weight)[:selected_feature_num]
                elif Selector_name == 'UDFS':
                    Weight = UDFS.udfs(X_train, gamma=0.1, n_clusters=2)
                    idx = feature_ranking(Weight)[:selected_feature_num]
                elif Selector_name == 'CFS':
                    idx = CFS.cfs(X_train, y_train)
                elif Selector_name == 'f_score':
                    score = f_score.f_score(X_train, y_train)
                    idx = f_score.feature_ranking(score)[:selected_feature_num]
                elif Selector_name == 'gini_index':
                    score = gini_index.gini_index(X_train, y_train)
                    idx = gini_index.feature_ranking(score)[:selected_feature_num]
                elif Selector_name == 't_score':
                    score = t_score.t_score(X_train, y_train)
                    idx = t_score.feature_ranking(score)[:selected_feature_num]
                elif Selector_name == 'alpha_investing':
                    try:
                        idx = alpha_investing.alpha_investing(X_train, y_train, 0.5, 0.5)
                    except:
                        idx = random.sample([i for i in range(len(feature_name_list))], selected_feature_num)
                elif Selector_name == 'LASSO':
                    alphas = np.logspace(-3, 1, 100)
                    find_sign = 1
                    for alpha in alphas:
                        model_lasso = LassoCV(alphas=[alpha], cv=10, max_iter=10000, random_state=seed).fit(X_train, y_train)
                        coef = pd.Series(model_lasso.coef_, index=feature_name_list)
                        if sum(coef != 0) == selected_feature_num:
                            coef_ = model_lasso.coef_
                            idx = np.where(coef_ != 0)[0].tolist()
                            find_sign = 0
                            break
                    if find_sign:
                        idx = random.sample([i for i in range(len(feature_name_list))], selected_feature_num)

                features_name_Selectored = [feature_name_list[i] for i in idx]
                remove_features_name = list(set(feature_name_list) - set(features_name_Selectored))
                df_train_Selectored.drop(columns=remove_features_name, inplace=True)
                df_val_Selectored.drop(columns=remove_features_name, inplace=True)
                df_external1_Selectored.drop(columns=remove_features_name, inplace=True)
                df_prospective_Selectored.drop(columns=remove_features_name, inplace=True)
                df_train_Selectored.to_excel(os.path.join(Selector_dir, 'train_data_Selectored.xlsx'), index=False)
                df_val_Selectored.to_excel(os.path.join(Selector_dir, 'val_data_Selectored.xlsx'), index=False)
                df_external1_Selectored.to_excel(os.path.join(Selector_dir, 'external1_data_Selectored.xlsx'), index=False)
                df_prospective_Selectored.to_excel(os.path.join(Selector_dir, 'prospective_data_Selectored.xlsx'), index=False)

            train_ID_list = list(df_train_Selectored['unique_id'])
            X_train = np.array(df_train_Selectored.iloc[:, 2:].values)
            Y_train = np.array(df_train_Selectored['Group'])
            val_ID_list = list(df_val_Selectored['unique_id'])
            X_val = np.array(df_val_Selectored.iloc[:, 2:].values)
            Y_val = np.array(df_val_Selectored['Group'])
            external1_ID_list = list(df_external1_Selectored['unique_id'])
            X_external1 = np.array(df_external1_Selectored.iloc[:, 2:].values)
            Y_external1 = np.array(df_external1_Selectored['Group'])
            prospective_ID_list = list(df_prospective_Selectored['unique_id'])
            X_prospective = np.array(df_prospective_Selectored.iloc[:, 2:].values)
            Y_prospective = np.array(df_prospective_Selectored['Group'])
            for Classifier_name in ['Logistic_Regression', 'KNN', 'SVM', 'Naive_Bayes', 'Decision_Tree', 'Extra_Trees',
                                    'Random_Forest', 'AdaBoost', 'GradientBoosting', 'XGBoost', 'LightGBM',
                                    'Catboost', 'ANN']:
                print('{} {} {} {}'.format(Imputer_name, Normalizer_name, Selector_name, Classifier_name))
                Classifier_dir = os.path.join(Selector_dir, Classifier_name)
                os.makedirs(Classifier_dir, exist_ok=True)
                if Classifier_name == 'Logistic_Regression':
                    Grid_Dict = {'solver': ['liblinear', 'lbfgs', 'sag']}
                    clf = LogisticRegression(penalty='l2', random_state=None)
                    Classifier = GridSearchCV(clf, cv=5, param_grid=Grid_Dict)
                    Classifier.fit(X_train, Y_train)
                    y_predict_train = Classifier.predict_proba(X_train)[:, 1]
                    y_predict_val = Classifier.predict_proba(X_val)[:, 1]
                    y_predict_external1 = Classifier.predict_proba(X_external1)[:, 1]
                    y_predict_prospective = Classifier.predict_proba(X_prospective)[:, 1]
                elif Classifier_name == 'KNN':
                    Grid_Dict = {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance']}
                    clf = KNeighborsClassifier(n_jobs=4)
                    Classifier = GridSearchCV(clf, cv=5, param_grid=Grid_Dict)
                    Classifier.fit(X_train, Y_train)
                    y_predict_train = Classifier.predict_proba(X_train)[:, 1]
                    y_predict_val = Classifier.predict_proba(X_val)[:, 1]
                    y_predict_external1 = Classifier.predict_proba(X_external1)[:, 1]
                    y_predict_prospective = Classifier.predict_proba(X_prospective)[:, 1]
                elif Classifier_name == 'SVM':
                    Grid_Dict = {'kernel': ['linear', 'rbf']}
                    clf = SVC(probability=True, C=1)
                    Classifier = GridSearchCV(clf, cv=5, param_grid=Grid_Dict)
                    Classifier.fit(X_train, Y_train)
                    y_predict_train = Classifier.predict_proba(X_train)[:, 1]
                    y_predict_val = Classifier.predict_proba(X_val)[:, 1]
                    y_predict_external1 = Classifier.predict_proba(X_external1)[:, 1]
                    y_predict_prospective = Classifier.predict_proba(X_prospective)[:, 1]
                elif Classifier_name == 'Naive_Bayes':
                    Grid_Dict = {}
                    clf = GaussianNB()
                    Classifier = GridSearchCV(clf, cv=5, param_grid=Grid_Dict)
                    Classifier.fit(X_train, Y_train)
                    y_predict_train = Classifier.predict_proba(X_train)[:, 1]
                    y_predict_val = Classifier.predict_proba(X_val)[:, 1]
                    y_predict_external1 = Classifier.predict_proba(X_external1)[:, 1]
                    y_predict_prospective = Classifier.predict_proba(X_prospective)[:, 1]
                elif Classifier_name == 'Decision_Tree':
                    Grid_Dict = {}
                    clf = DecisionTreeClassifier()
                    Classifier = GridSearchCV(clf, cv=5, param_grid=Grid_Dict)
                    Classifier.fit(X_train, Y_train)
                    y_predict_train = Classifier.predict_proba(X_train)[:, 1]
                    y_predict_val = Classifier.predict_proba(X_val)[:, 1]
                    y_predict_external1 = Classifier.predict_proba(X_external1)[:, 1]
                    y_predict_prospective = Classifier.predict_proba(X_prospective)[:, 1]
                elif Classifier_name == 'Extra_Trees':
                    Grid_Dict = {}
                    clf = ExtraTreesClassifier(n_jobs=-1)
                    Classifier = GridSearchCV(clf, cv=5, param_grid=Grid_Dict)
                    Classifier.fit(X_train, Y_train)
                    y_predict_train = Classifier.predict_proba(X_train)[:, 1]
                    y_predict_val = Classifier.predict_proba(X_val)[:, 1]
                    y_predict_external1 = Classifier.predict_proba(X_external1)[:, 1]
                    y_predict_prospective = Classifier.predict_proba(X_prospective)[:, 1]
                elif Classifier_name == 'Random_Forest':
                    Grid_Dict = {'n_estimators': [10, 50, 100, 200], 'max_depth': [1, 5, 10, 20]}
                    clf = RandomForestClassifier(class_weight='balanced', n_jobs=-1)
                    Classifier = GridSearchCV(clf, cv=5, param_grid=Grid_Dict)
                    Classifier.fit(X_train, Y_train)
                    y_predict_train = Classifier.predict_proba(X_train)[:, 1]
                    y_predict_val = Classifier.predict_proba(X_val)[:, 1]
                    y_predict_external1 = Classifier.predict_proba(X_external1)[:, 1]
                    y_predict_prospective = Classifier.predict_proba(X_prospective)[:, 1]
                elif Classifier_name == 'AdaBoost':
                    Grid_Dict = {'n_estimators': [1, 10, 50, 100, 200]}
                    clf = AdaBoostClassifier()
                    Classifier = GridSearchCV(clf, cv=5, param_grid=Grid_Dict)
                    Classifier.fit(X_train, Y_train)
                    y_predict_train = Classifier.predict_proba(X_train)[:, 1]
                    y_predict_val = Classifier.predict_proba(X_val)[:, 1]
                    y_predict_external1 = Classifier.predict_proba(X_external1)[:, 1]
                    y_predict_prospective = Classifier.predict_proba(X_prospective)[:, 1]
                elif Classifier_name == 'GradientBoosting':
                    Grid_Dict = {'n_estimators': [1, 10, 50, 100, 200], 'loss': ['exponential', 'log_loss']}
                    clf = GradientBoostingClassifier()
                    Classifier = GridSearchCV(clf, cv=5, param_grid=Grid_Dict)
                    Classifier.fit(X_train, Y_train)
                    y_predict_train = Classifier.predict_proba(X_train)[:, 1]
                    y_predict_val = Classifier.predict_proba(X_val)[:, 1]
                    y_predict_external1 = Classifier.predict_proba(X_external1)[:, 1]
                    y_predict_prospective = Classifier.predict_proba(X_prospective)[:, 1]
                elif Classifier_name == 'XGBoost':
                    Grid_Dict = {'n_estimators': [1, 10, 50, 100, 200], 'max_depth': [1, 5, 10, 20], 'learning_rate': [0.01, 0.1]}
                    clf = XGBClassifier(booster='gbtree', objective='binary:logistic', eval_metric='logloss', n_jobs=-1)
                    Classifier = GridSearchCV(clf, cv=5, param_grid=Grid_Dict)
                    Classifier.fit(X_train, Y_train)
                    y_predict_train = Classifier.predict_proba(X_train)[:, 1]
                    y_predict_val = Classifier.predict_proba(X_val)[:, 1]
                    y_predict_external1 = Classifier.predict_proba(X_external1)[:, 1]
                    y_predict_prospective = Classifier.predict_proba(X_prospective)[:, 1]
                elif Classifier_name == 'LightGBM':
                    Grid_Dict = {}
                    clf = LGBMClassifier(n_jobs=-1)
                    Classifier = GridSearchCV(clf, cv=5, param_grid=Grid_Dict)
                    Classifier.fit(X_train, Y_train)
                    y_predict_train = Classifier.predict_proba(X_train)[:, 1]
                    y_predict_val = Classifier.predict_proba(X_val)[:, 1]
                    y_predict_external1 = Classifier.predict_proba(X_external1)[:, 1]
                    y_predict_prospective = Classifier.predict_proba(X_prospective)[:, 1]
                elif Classifier_name == 'Catboost':
                    Grid_Dict = {}
                    clf = CatBoostClassifier(logging_level='Silent')
                    Classifier = GridSearchCV(clf, cv=5, param_grid=Grid_Dict)
                    Classifier.fit(X_train, Y_train)
                    y_predict_train = Classifier.predict_proba(X_train)[:, 1]
                    y_predict_val = Classifier.predict_proba(X_val)[:, 1]
                    y_predict_external1 = Classifier.predict_proba(X_external1)[:, 1]
                    y_predict_prospective = Classifier.predict_proba(X_prospective)[:, 1]
                elif Classifier_name == 'ANN':
                    model, device = train_ANN(X_train, Y_train)
                    X_train_tensor = torch.from_numpy(X_train).to(device).float()
                    X_val_tensor = torch.from_numpy(X_val).to(device).float()
                    X_external1_tensor = torch.from_numpy(X_external1).to(device).float()
                    X_prospective_tensor = torch.from_numpy(X_prospective).to(device).float()
                    with torch.no_grad():
                        y_predict_train = torch.softmax(model(X_train_tensor), dim=1).cpu().numpy()[:, 1]
                        y_predict_val = torch.softmax(model(X_val_tensor), dim=1).cpu().numpy()[:, 1]
                        y_predict_external1 = torch.softmax(model(X_external1_tensor), dim=1).cpu().numpy()[:, 1]
                        y_predict_prospective = torch.softmax(model(X_prospective_tensor), dim=1).cpu().numpy()[:, 1]
                    torch.save(model.state_dict(), os.path.join(Classifier_dir, 'model.pth'))
                # AUC_train, ACC_train, Specifcity_train, Sensitivity_train, F1_train = metric(Y_train, y_predict_train)
                # AUC_val, ACC_val, Specifcity_val, Sensitivity_val, F1_val = metric(Y_val, y_predict_val)
                # AUC_external1, ACC_external1, Specifcity_external1, Sensitivity_external1, F1_external1 = metric(Y_external1, y_predict_external1)
                # AUC_prospective, ACC_prospective, Specifcity_prospective, Sensitivity_prospective, F1_prospective = metric(Y_prospective, y_predict_prospective)
                # print('train: {} {} {} {} {}'.format(AUC_train, ACC_train, Specifcity_train, Sensitivity_train, F1_train))
                # print('val: {} {} {} {} {}'.format(AUC_val, ACC_val, Specifcity_val, Sensitivity_val, F1_val))
                # print('external1: {} {} {} {} {}'.format(AUC_external1, ACC_external1, Specifcity_external1, Sensitivity_external1, F1_external1))
                # print('prospective: {} {} {} {} {}'.format(AUC_prospective, ACC_prospective, Specifcity_prospective, Sensitivity_prospective, F1_prospective))
                if not Classifier_name == 'ANN':
                    joblib.dump(Classifier, os.path.join(Classifier_dir, 'model.pkl'))
                df_train_score = pd.DataFrame({'ID': train_ID_list, 'label': Y_train.tolist(), 'score': y_predict_train.tolist()})
                df_val_score = pd.DataFrame({'ID': val_ID_list, 'label': Y_val.tolist(), 'score': y_predict_val.tolist()})
                df_external1_score = pd.DataFrame({'ID': external1_ID_list, 'label': Y_external1.tolist(), 'score': y_predict_external1.tolist()})
                df_prospective_score = pd.DataFrame({'ID': prospective_ID_list, 'label': Y_prospective.tolist(), 'score': y_predict_prospective.tolist()})
                df_train_score.to_excel(os.path.join(Classifier_dir, 'train_score.xlsx'), index=False)
                df_val_score.to_excel(os.path.join(Classifier_dir, 'val_score.xlsx'), index=False)
                df_external1_score.to_excel(os.path.join(Classifier_dir, 'external1_score.xlsx'), index=False)
                df_prospective_score.to_excel(os.path.join(Classifier_dir, 'prospective_score.xlsx'), index=False)