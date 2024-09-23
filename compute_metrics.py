import os
import numpy as np
import pandas as pd
from utils import metric_get_threshold

save_dir = 'trained_models/center0234_center1_center5'
use_feature_mode = 'all'
seed = 0

df = pd.DataFrame(columns=['Imputer', 'Normalizer', 'Selector', 'Classifier', 'Cut_off',
                           'train_AUC', 'train_ACC', 'train_Specifcity', 'train_Sensitivity', 'train_F1',
                           'val_AUC', 'val_ACC', 'val_Specifcity', 'val_Sensitivity', 'val_F1',
                           'external1_AUC', 'external1_ACC', 'external1_Specifcity', 'external1_Sensitivity', 'external1_F1',
                           'prospective_AUC', 'prospective_ACC', 'prospective_Specifcity', 'prospective_Sensitivity', 'prospective_F1'])

for Imputer_name in ['IterativeImputer', 'MeanImputer', 'MedianImputer', 'KNNImputer']:
    for Normalizer_name in ['Yeo_Johnson', 'MinMax', 'Standard']:
        for Selector_name in ['CIFE', 'CMIM', 'DISR', 'ICAP', 'JMI', 'MIFS', 'MIM', 'MRMR', 'fisher_score',
                              'lap_score', 'reliefF', 'SPEC', 'trace_ratio', 'll_l21', 'ls_l21', 'MCFS',
                              'RFS', 'UDFS', 'CFS', 'f_score', 'gini_index', 't_score', 'alpha_investing',
                              'LASSO']:
            for Classifier_name in ['Logistic_Regression', 'KNN', 'SVM', 'Naive_Bayes', 'Decision_Tree', 'Extra_Trees',
                                    'Random_Forest', 'AdaBoost', 'GradientBoosting', 'XGBoost', 'LightGBM',
                                    'Catboost', 'ANN']:
                print('{} {} {} {}'.format(Imputer_name, Normalizer_name, Selector_name, Classifier_name))
                df_train_score = pd.read_excel(os.path.join(save_dir, 'use_{}_features'.format(use_feature_mode),
                                                            'train_val_seed{}'.format(seed), Imputer_name,
                                                            Normalizer_name, Selector_name, Classifier_name,
                                                            'train_score.xlsx'))
                df_val_score = pd.read_excel(os.path.join(save_dir, 'use_{}_features'.format(use_feature_mode),
                                                          'train_val_seed{}'.format(seed), Imputer_name,
                                                            Normalizer_name, Selector_name, Classifier_name,
                                                            'val_score.xlsx'))
                df_external1_score = pd.read_excel(os.path.join(save_dir, 'use_{}_features'.format(use_feature_mode),
                                                          'train_val_seed{}'.format(seed), Imputer_name,
                                                          Normalizer_name, Selector_name, Classifier_name,
                                                          'external1_score.xlsx'))
                df_prospective_score = pd.read_excel(os.path.join(save_dir, 'use_{}_features'.format(use_feature_mode),
                                                                'train_val_seed{}'.format(seed), Imputer_name,
                                                                Normalizer_name, Selector_name, Classifier_name,
                                                                'prospective_score.xlsx'))
                Y_train = np.array(df_train_score['label'])
                y_predict_train = np.array(df_train_score['score'])
                Y_val = np.array(df_val_score['label'])
                y_predict_val = np.array(df_val_score['score'])
                Y_external1 = np.array(df_external1_score['label'])
                y_predict_external1 = np.array(df_external1_score['score'])
                Y_prospective = np.array(df_prospective_score['label'])
                y_predict_prospective = np.array(df_prospective_score['score'])
                AUC_train, ACC_train, Specifcity_train, Sensitivity_train, F1_train, threshold = metric_get_threshold(Y_train, y_predict_train, threshold_=None)
                AUC_val, ACC_val, Specifcity_val, Sensitivity_val, F1_val, _ = metric_get_threshold(Y_val, y_predict_val, threshold_=threshold)
                AUC_external1, ACC_external1, Specifcity_external1, Sensitivity_external1, F1_external1, _ = metric_get_threshold(Y_external1, y_predict_external1, threshold_=threshold)
                AUC_prospective, ACC_prospective, Specifcity_prospective, Sensitivity_prospective, F1_prospective, _ = metric_get_threshold(Y_prospective, y_predict_prospective, threshold_=threshold)
                print('train: {} {} {} {} {}'.format(AUC_train, ACC_train, Specifcity_train, Sensitivity_train, F1_train))
                print('val: {} {} {} {} {}'.format(AUC_val, ACC_val, Specifcity_val, Sensitivity_val, F1_val))
                print('external1: {} {} {} {} {}'.format(AUC_external1, ACC_external1, Specifcity_external1, Sensitivity_external1, F1_external1))
                print('prospective: {} {} {} {} {}'.format(AUC_prospective, ACC_prospective, Specifcity_prospective, Sensitivity_prospective, F1_prospective))
                df.loc[len(df)] = [Imputer_name, Normalizer_name, Selector_name, Classifier_name, threshold,
                                          AUC_train, ACC_train, Specifcity_train, Sensitivity_train, F1_train,
                                          AUC_val, ACC_val, Specifcity_val, Sensitivity_val, F1_val,
                                        AUC_external1, ACC_external1, Specifcity_external1, Sensitivity_external1, F1_external1,
                                   AUC_prospective, ACC_prospective, Specifcity_prospective, Sensitivity_prospective, F1_prospective]
df.to_excel(os.path.join(save_dir, 'use_{}_features'.format(use_feature_mode), 'train_val_seed{}'.format(seed), 'all_metrics.xlsx'), index=False)