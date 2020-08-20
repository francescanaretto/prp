import pickle
import itertools
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
import numpy as np
from sklearn.utils import class_weight
from sklearn import metrics
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold

class Logistic_Regression:
    def __init__(self, labels=None, dataset=None, title=None):
        self.labels = labels
        self.dataset = dataset
        self.title = title

    #save the datasets in pickle files
    def train_test_stratified(self):
        train_set, test_set, train_label, test_label = train_test_split(self.dataset, self.labels, stratify=self.labels,test_size=0.25)
        title = "../datasets/train_set.p"
        with open(title, 'wb') as fp:
            pickle.dump(train_set, fp)
        title = "../datasets/train_label.p"
        with open(title, 'wb') as fp:
            pickle.dump(train_label, fp)
        title = "../datasets/test_set.p"
        with open(title, 'wb') as fp:
            pickle.dump(test_set, fp)
        title = "../datasets/test_label.p"
        with open(title, 'wb') as fp:
            pickle.dump(test_label, fp)

    #function to load the trained model and the history, useful for evaluation purposes
    def load_model(self, title_model, title_history, path):
        trained_model = open(title_model,"rb")
        self.model = pickle.load(trained_model)
        model_history = open(title_history,"rb")
        self.history = pickle.load(model_history)
        title = "../datasets/"+path+"/test_set.p"
        test = open(title,"rb")
        self.test_set = pickle.load(test)
        title = "../datasets/"+path+"/test_label.p"
        test_l = open(title,"rb")
        self.test_label = pickle.load(test_l)

    def train_validation_undersampling(self, train_set, train_label, sampling_strategy):
        self.train_set, self.validation_set, self.train_label, self.validation_label = train_test_split(train_set, train_label, stratify=train_label,test_size=0.15)
        sampler = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)
        self.train_set, self.train_label = sampler.fit_sample(self.train_set, self.train_label)
        print(self.train_set.shape)
        print(self.train_label.shape)
        title = "../datasets/train_under_set.p"
        with open(title, 'wb') as fp:
            pickle.dump(self.train_set, fp)
        title = "../datasets/train_under_label.p"
        with open(title, 'wb') as fp:
            pickle.dump(self.train_label, fp)


    def train_validation_raw(self, train_set, train_label):
        self.train_set, self.validation_set, self.train_label, self.validation_label = train_test_split(train_set, train_label, stratify=train_label, test_size=0.15)
        title = "../datasets/train_raw_set.p"
        with open(title, 'wb') as fp:
            pickle.dump(self.train_set, fp)
        title = "../datasets/train_raw_label.p"
        with open(title, 'wb') as fp:
            pickle.dump(self.train_label, fp)
        title = "../datasets/validation_raw_set.p"
        with open(title, 'wb') as fp:
            pickle.dump(self.validation_set, fp)
        title = "../datasets/validation_raw_label.p"
        with open(title, 'wb') as fp:
            pickle.dump(self.validation_label, fp)

    def k_fold_validation(self, k, train_set, train_label, weight_flag):
        kf = StratifiedKFold(n_splits=k,shuffle=True,random_state=42)
        count = 0
        scores = dict()
        train_set = train_set.values
        train_label = train_label.values
        for train_index,test_index in kf.split(train_set, train_label):
            train, test = train_set[train_index], train_set[test_index]
            t_label, test_label = train_label[train_index], train_label[test_index]

            #model
            model = self.logistic_regression_model(weight_flag)
            model = self.logistic_regression_fit(model, train, t_label)
            #saving the model into a pickle file
            title_model = '../results/trained_lg_'+str(self.title)+str(count)+'.p'
            pickle.dump(model, open(title_model, 'wb'))
            scores[count] = self.logistic_regression_prediction(model, test, test_label)
            write_index = open('index_'+self.title+str(count)+'.txt',"w")
            write_index.write(str(train_index))
            write_index.write(str(test_index))
            count += 1
        print(scores)
        scores_history = '../results/scores'+str(self.title)+'.p'
        pickle.dump(scores, open(scores_history, 'wb'))


    def logistic_regression_model(self, weight_flag=False):
        if weight_flag is True:
            #fit the model
            model = LogisticRegression(C=0.09, penalty= 'l2')
        else:
            model = LogisticRegression(C= 0.09, penalty='l2')
        return model

    def grid_search_parameters(self):
        grid={"C":np.logspace(-3,3,7), "penalty":["l1","l2"]}
        logreg=LogisticRegression()
        logreg_cv=GridSearchCV(logreg,grid,cv=10)
        print(logreg.get_params())
        return logreg_cv


    def logistic_regression_fit(self, model, train_set, train_label):
        model = model.fit(train_set, train_label)
        return model

    def logistic_regression_prediction(self, model, test_set, test_label):
        self.predictions = model.predict(test_set)
        score = model.score(test_set, test_label)
        report = classification_report(test_label, self.predictions)
        cm = metrics.confusion_matrix(test_label, self.predictions)
        self.plot_confusion_matrix_multi(cm, ['Low risk', 'High risk'])
        title = "../results/report_"+self.title+".txt"
        write_report = open(title,"w")
        write_report.write(report)

        print(report)
        return report



    def plot_confusion_matrix_multi(self, cm,target_names, cmap=None, normalize=True):
        accuracy = np.trace(cm) / float(np.sum(cm))
        misclass = 1 - accuracy
        if cmap is None:
            cmap = plt.get_cmap('Blues')
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title('Confusion Matrix for the Logistic Regression for mobility data \n Stratified sample \n', size=16)
        plt.colorbar()
        if target_names is not None:
            tick_marks = np.arange(len(target_names))
            plt.xticks(tick_marks, target_names, size=14)
            plt.yticks(tick_marks, target_names, size=14)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        thresh = cm.max() / 1.5 if normalize else cm.max() / 2
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if normalize:
                plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="grey" if cm[i, j] > thresh else "black", size=14)
            else:
                plt.text(j, i, "{:,}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="grey" if cm[i, j] > thresh else "black", size=14)
        plt.tight_layout()
        plt.ylabel('True label', size=14)
        plt.gcf().subplots_adjust(bottom=0.20)
        plt.xlabel('\n Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass), size=14)
        plt.savefig('../results/cm_'+str(self.title)+'.png')



    def measures(self, test_label):
        report = classification_report(test_label, self.predictions)
        print(report)
        write_report = open('measures_'+self.title+'.txt',"w")
        write_report.write(report)
