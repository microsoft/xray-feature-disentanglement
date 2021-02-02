'''
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
'''
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

import utils

def do_experiment(embeddings, labels, num_splits=10, seed=1, number_of_classes=5, verbose=False):
    accs = []
    aucs = []

    kf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=seed)
    for split_idx, (train_index, test_index) in enumerate(kf.split(embeddings, labels)):
        if verbose:
            print("%d/%d" % (split_idx, num_splits))
        x_train, y_train = embeddings[train_index], labels[train_index]
        x_test, y_test = embeddings[test_index], labels[test_index]
        
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test  = scaler.transform(x_test)
            
        model = LogisticRegression(C=1.0, multi_class="multinomial", class_weight=None, max_iter=2000)
        model.fit(x_train, y_train)

        y_pred_proba = model.predict_proba(x_test)
        
        accs.append(utils.get_per_class_accuracies(
            y_test, y_pred_proba.argmax(axis=1), number_classes=number_of_classes
        ))
        aucs.append(utils.get_binary_aucs(y_test, y_pred_proba, number_classes=number_of_classes))
        
    accs = np.array(accs)
    aucs = np.array(aucs)
    average_aucs = np.mean(aucs, axis=1)
    average_accs = np.mean(accs, axis=1)
    results = {
        "average auc": (np.mean(average_aucs), np.std(average_aucs)),
        "average acc": (np.mean(average_accs), np.std(average_accs)),
    }
    return results


models = ["xrv", "histogram", "densenet", "covidnet"]
masks = ["masked", "unmasked"]

print("Linear model performance discriminating between sub-datasets in the COVIDx dataset from pre-trained embeddings")
print("")
print("Masking method,Feature extractor model,Average AUC,Average ACC")
for mask in masks:
    for model in models:
        embeddings = utils.get_embeddings("covidx", mask, model)
        labels = utils.get_domain_labels("covidx")
        results = do_experiment(embeddings, labels, number_of_classes=5)
        print("%s,%s,%0.2f +/- %0.2f,%0.2f +/- %0.2f" % (
            mask, model,
            results["average auc"][0], results["average auc"][1],
            results["average acc"][0], results["average acc"][1],
        ))