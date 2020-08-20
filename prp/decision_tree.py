from sklearn import tree
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

print("init")
name = 'vor3'
data = open("../../datasets/features_mobility_voronoi.p","rb")
dataset = pickle.load(data)
risk = pd.read_csv('../../datasets/rischio'+name+'.csv')

dataset = pd.merge(dataset, risk, on=('uid'))
print(risk['uid'])


data = dataset.pop('uid')
title = "../../datasets/train_set_"+name+".p"
train = open(title,"rb")
train_set = pickle.load(train)
title = "../../datasets/train_label_"+name+".p"
train_l = open(title,"rb")
train_label = pickle.load(train_l)
title = "../../datasets/test_set_"+name+".p"
test = open(title,"rb")
test_set = pickle.load(test)
title = "../../datasets/test_label_"+name+".p"
test_l = open(title,"rb")
test_label = pickle.load(test_l)

classifier = tree.DecisionTreeClassifier(criterion= 'gini', max_depth= 20, max_features= 'sqrt', min_samples_leaf= 10, min_samples_split= 40)
classifier.fit(train_set, train_label)
title_model = '../results/trained_dt'+name+'.p'
pickle.dump(classifier, open(title_model, 'wb'))
y_pred = classifier.predict(test_set)
acc = accuracy_score(test_label, y_pred)
print("Test Accuracy of DT = {:.2f} %".format(acc * 100))
print(acc)
report = classification_report(test_label, y_pred)
title = "report_dt_"+name+".txt"
write_report = open(title,"w")
write_report.write(report)
