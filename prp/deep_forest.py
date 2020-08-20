"""
MNIST datasets demo for gcforest
Usage:
    define the model within scripts:
        python examples/demo_mnist.py
    get config from json file:
        python examples/demo_mnist.py --model examples/demo_mnist-gc.json
        python examples/demo_mnist.py --model examples/demo_mnist-ca.json
"""
import argparse
import numpy as np
import sys
import pickle
from sklearn.metrics import accuracy_score
sys.path.insert(0, "lib")
from sklearn.metrics import classification_report
from gcforest.gcforest import GCForest


def get_config():
    config = {}
    ca_config = {}
    ca_config["random_state"] = 42
    ca_config["max_layers"] = 3000
    ca_config["early_stopping_rounds"] = 5
    ca_config["n_classes"] = 2
    ca_config["estimators"] = []
    ca_config["estimators"].append({'n_folds':1, 'type': 'RandomForestClassifier', 'bootstrap': True,'criterion': 'gini','max_depth': 100,'max_features': 'sqrt','min_samples_leaf': 5,'min_samples_split': 25,'n_estimators': 30})
    ca_config["estimators"].append({"n_folds": 50, "type": 'XGBClassifier', 'bootstrap': True,'eta': 0,'gamma': 8,'learning_rate': 0.01,'max_depth': 20,'n_estimators': 300,'tree_method': 'approx', "n_jobs": 6})
    ca_config["estimators"].append({"n_folds": 1, "type": "ExtraTreesClassifier", 'bootstrap': False,'criterion': 'gini','max_depth': 500,'max_features': 'sqrt','min_samples_leaf': 3,'min_samples_split': 10,'n_estimators': 30})
    ca_config["estimators"].append({'n_folds': 1, 'type': 'RandomForestClassifier','bootstrap': True,'criterion': 'entropy','max_depth': 500,'max_features': 'auto','min_samples_leaf': 3,'min_samples_split': 10,'n_estimators': 200})
    config["cascade"] = ca_config
    return config


if __name__ == "__main__":
    name = 'vor3'
    pickle_in = open('../../datasets/train_set_'+name+'.p',"rb")
    train_set = pickle.load(pickle_in)
    pickle_in = open('../../datasets/train_label_'+name+'.p',"rb")
    train_label = pickle.load(pickle_in)
    pickle_in = open('../../datasets/test_set_'+name+'.p', "rb")
    test_set = pickle.load(pickle_in)
    pickle_in = open('../../datasets/test_label_'+name+'.p', "rb")
    test_label = pickle.load(pickle_in)
    train_set = train_set.to_numpy()
    train_label = train_label.to_numpy()
    test_set = test_set.to_numpy()
    test_label = test_label.to_numpy()
    print(len(np.unique(train_label)), len(np.unique(train_set)))
    print(train_label.shape, train_set.shape)
    config = get_config()
    gc = GCForest(config)
    print(config)
    X_train_enc = gc.fit_transform(train_set, train_label)
    title_model = '../results/trained_deep_forest_'+name+'.p'
    pickle.dump(gc, open(title_model, 'wb'))
    y_pred = gc.predict(test_set)
    acc = accuracy_score(test_label, y_pred)
    print("Test Accuracy of GcForest = {:.2f} %".format(acc * 100))
    print(acc)
    report = classification_report(test_label, y_pred)
    title = "report_deep_forest_"+name+".txt"
    write_report = open(title,"w")
    write_report.write(report)
