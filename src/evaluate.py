import os
import pandas as pd
import numpy as np
import argparse
import json
import pickle

from utils import load_and_preprocess_spectra
from train import tune_model

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, f1_score, confusion_matrix, log_loss, precision_score, recall_score


def main():
    np.random.seed(222)

    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help='Config file path')
    parser.add_argument("antimicrobial", 
                        type=str, 
                        help='Antimicrobial')
    parser.add_argument("model", 
                        type=str, 
                        help='Model type. Currently support LR, RF, SVM, XGB, MLP')

    args = parser.parse_args()
    antimicrobial = args.antimicrobial
    model = args.model
    model_path = os.path.join(args["result_dir"], 
                              f"best_model_{antimicrobial}_{model}.pkl")
    result_path = os.path.join(args["result_dir"], 
                               f"result_{antimicrobial}_{model}.txt")

    # Load config file
    with open(args.config, 'r') as config_input:
        config = json.load(config_input)

    # Loading ids
    ids = pd.read_csv(config["id_path"], header=None)\
                  .values.flatten()
    
    # Loading AST label
    ast_df = pd.read_csv(config["ast_label_path"])

    if config["bins_path"]:
        # Load bins
        with open(config["bins_path"], 'rb') as bins_input:
            features = pickle.load(bins_input)

        if type(features) == np.ndarray:
            pass
        else:
            features = features.values
            
        bins = features.astype('int')
        bins = np.append(bins, [20000]) # Adding the righmost edge

        # Load training and testing dataset
        X_val, y_val = load_and_preprocess_spectra(config["input_dir"],
                                                    ast_df,
                                                    antimicrobial,
                                                    ids, 
                                                    bins)
    else:
        X = pd.read_csv(config['input_dir'])
        features=list(X.columns)
        features.remove('id')
        
        X_val = X.merge(pd.DataFrame({'id': ids}),
                          how='inner',
                          on='id')\
                   .drop(labels=['id'], axis=1)\
                   .to_numpy()
        y_val = ast_df.merge(pd.DataFrame({'id': ids}),
                             how='inner',
                             on='id')[antimicrobial].values

    # Normalization
    norm_coeff = config["normal_coeff"]
    X_val = np.clip(X_val / norm_coeff, 0, 1)

    # Adding extra information
    if config['extra_features_path']:
        extra_feature_df = pd.read_csv(config['extra_features_path']) # Don't forget one-hot encoding and scaling!
        
        extra_features = list(extra_feature_df.columns)
        extra_features.remove('id')
        features = np.concatenate((features, extra_features))

        extra_feature_df = extra_feature_df.merge(pd.DataFrame({'id': ids}), on='id', how='inner').drop(labels=['id'], axis=1).to_numpy()

        X_val = np.concatenate([X_val, extra_feature_df], axis=1)

    # Load model and optimal threshold
    with open(model_path, "rb") as pretrained_model_input:
        clf = pickle.load(pretrained_model_input)
    
    performance_df = pd.read_csv(result_path, header=0, delimiter="\t")
    optimal_threshold = performance_df['threshold'].values[0]

    y_proba = clf.predict_proba(X_val)[:, 1]
    y_pred = (y_proba > optimal_threshold).astype('int')

    result_dict = {}
    result_dict['bacc'] = [np.round(balanced_accuracy_score(y_val, y_pred), 2)]
    result_dict['acc'] = [np.round(accuracy_score(y_val, y_pred), 2)]
    result_dict['auc'] = [np.round(roc_auc_score(y_val, y_proba), 2)]
    result_dict['f1'] = [np.round(f1_score(y_val, y_pred), 2)]
    result_dict['f1_macro'] = [np.round(f1_score(y_val, y_pred, average='macro'), 2)]
    result_dict['precision'] = [np.round(precision_score(y_val, y_pred), 2)]
    result_dict['recall'] = [np.round(recall_score(y_val, y_pred), 2)]
    result_dict['logloss'] = [np.round(log_loss(y_val, y_proba), 2)]
    tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
    result_dict['tn'] = [tn]
    result_dict['fp'] = [fp]
    result_dict['fn'] = [fn]
    result_dict['tp'] = [tp]

    result_path = os.path.join(config["output_dir"], 
                               f"result_{antimicrobial}_{model}.txt")

    pd.DataFrame(result_dict).to_csv(result_path, index=False, sep='\t')

    result_df = pd.DataFrame()
    result_df['y_val'] = y_val
    result_df['y_proba'] = y_proba
    result_df['y_pred'] = y_pred
    result_df.to_csv(os.path.join(config["output_dir"], f"result_{antimicrobial}_{model}.csv"), index=False)

if __name__ == "__main__":
    main()