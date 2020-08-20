import pickle
import itertools
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.utils import class_weight
from sklearn import metrics
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold

class Random_Forest_model:
    def __init__(self, labels=None, dataset=None, title=None):
        self.labels = labels
        self.dataset = dataset
        self.title = title

    #split train 80, test 20
    #save the datasets in pickle files
    def train_test_stratified(self):
        train_set, test_set, train_label, test_label = train_test_split(self.dataset, self.labels, stratify=self.labels,test_size=0.30)
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

    def train_validation_undersampling(self, train_set, train_label, sampling_strategy):
        self.train_set, self.validation_set, self.train_label, self.validation_label = train_test_split(train_set, train_label, stratify=train_label,test_size=0.15)
        # RandomUnderSampler
        sampler = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)
        self.train_set, self.train_label = sampler.fit_sample(self.train_set, self.train_label)
        print(self.train_set.shape)
        print(self.train_label.shape)
        sampler = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)
        self.validation_set, self.validation_label = sampler.fit_sample(self.validation_set, self.validation_label)
        print(self.validation_set.shape)
        print(self.validation_label.shape)
        title = "../datasets/train_under_set.p"
        with open(title, 'wb') as fp:
            pickle.dump(self.train_set, fp)
        title = "../datasets/train_under_label.p"
        with open(title, 'wb') as fp:
            pickle.dump(self.train_label, fp)
        title = "../datasets/validation_under_set.p"
        with open(title, 'wb') as fp:
            pickle.dump(self.validation_set, fp)
        title = "../datasets/validation_under_label.p"
        with open(title, 'wb') as fp:
            pickle.dump(self.validation_label, fp)


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


    def random_forest_model(self, weight_flag=False):
        if weight_flag is True:
            #fit the model
            model = RandomForestClassifier(class_weight='balanced', bootstrap= True, max_depth= 20, max_features= 5, min_samples_leaf= 15, min_samples_split= 30, n_estimators= 100)
        else:
            model = RandomForestClassifier(bootstrap=False, max_depth=20, max_features=4, min_samples_leaf=5,
                                           min_samples_split=10, n_estimators=400, criterion='entropy')
        return model


    def random_forest_fit(self, model, train_set, train_label):
        model = model.fit(train_set, train_label)
        #saving the model into a pickle file
        title_model = '../results/trained_rf_'+str(self.title)+'.p'
        pickle.dump(model, open(title_model, 'wb'))
        return model

    def random_forest_prediction(self, model, test_set, test_label):
        self.predictions = model.predict(test_set)
        score = model.score(test_set, test_label)
        report = classification_report(test_label, self.predictions)
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
        plt.title('Confusion Matrix for the Random Forest for mobility data \n Stratified sample \n', size=16)
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





    def confusion_matrix(self, test_label, target_names, cmap=None, normalize=True):
        cm = metrics.confusion_matrix(test_label, self.predictions)
        accuracy = np.trace(cm) / float(np.sum(cm))
        misclass = 1 - accuracy
        if cmap is None:
            cmap = plt.get_cmap('Blues')
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title('confusion_matrix'+self.title)
        plt.colorbar()
        if target_names is not None:
            tick_marks = np.arange(len(target_names))
            plt.xticks(tick_marks, target_names, rotation=45)
            plt.yticks(tick_marks, target_names)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        thresh = cm.max() / 1.5 if normalize else cm.max() / 2
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if normalize:
                plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
            else:
                plt.text(j, i, "{:,}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
        plt.savefig('../results/cm_'+str(self.title)+'.png')
