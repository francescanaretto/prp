import pickle
import shap
import pandas as pd
name = 'istat3'
title = "../datasets/train_set_"+name+".p"
train = open(title,"rb")
train_set = pickle.load(train)
title = "../datasets/train_label_"+name+".p"
train_l = open(title,"rb")
train_label = pickle.load(train_l)
title = "../datasets/test_set_"+name+".p"
test = open(title,"rb")
test_set = pickle.load(test)
title = "../datasets/test_label_"+name+".p"
test_l = open(title,"rb")
test_label = pickle.load(test_l)
pickle_in = open('../datasets/cv_trained_gcforest_'+name+'.p',"rb")
bb = pickle.load(pickle_in)
# use Kernel SHAP to explain test set predictions
explainer = shap.KernelExplainer(bb.predict_proba, test_set)
shap_values_list = list()
test_val = test_set.values
title = 'shap_values_'+name+'.p'
with open(title, "ab") as pickle_file:
    for t in test_val:
        shap_values = explainer.shap_values(t)
        shap_values_list.append(shap_values)
        pickle.dump(shap_values_list, pickle_file)


