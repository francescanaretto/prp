from lore.lorem import LOREM
from lore.datamanager import *
from lore.rule import apply_counterfactual
from lore.util import record2str, mixed_distance_idx, neuclidean, multilabel2str, nmeandev
import pickle

def main():

    #load dei dati
    name = 'istat2'
    pickle_in = open('../../../datasets/features_mobility.p',"rb")
    dataset = pickle.load(pickle_in)
    risk = pd.read_csv('../../../datasets/rischio'+name+'.csv', sep=r'\s*,\s*')
    dataset = pd.merge(dataset, risk, on=('uid'))
    data = dataset.pop('uid')
    class_name = 'label'
    df, feature_names, class_values, numeric_columns, rdf, real_feature_names, features_map = prepare_dataset(dataset, class_name)

    print(features_map)
    #load del modello di random forest
    pickle_in = open('../../../datasets/trained_deep_forest_'+name+'.p',"rb")
    bb = pickle.load(pickle_in)

    title = "../../../datasets/test_set_strat"+name+".p"
    test = open(title,"rb")
    test_set = pickle.load(test)
    if features_map:
        features_map_inv = dict()
        for idx, idx_dict in features_map.items():
            for k, v in idx_dict.items():
                features_map_inv[v] = idx

    idx_dc = len(numeric_columns)

    def mixed_distance(x, y):
        return mixed_distance_idx(x, y, idx_dc, ddist=neuclidean, cdist=neuclidean)
    def bb_predict(X):
        return bb.predict(X)
    def bb_predict_proba(X):
        return bb.predict_proba(X)

    test_val = test_set.values


    explainer = LOREM(test_val, bb_predict, feature_names, class_name, class_values, numeric_columns, features_map,
                      neigh_type='geneticp', categorical_use_prob=True,
                      continuous_fun_estimation=False, size=1000, ocr=0.1, multi_label=False, one_vs_rest=False,
                      filter_crules=False, random_state=0, verbose=True, ngen=10, bb_predict_proba=bb_predict_proba,
                      metric=neuclidean)
    explainations = list()
    log = open('lore_deep_forest'+name+'.txt','w')
    title = 'explanations_deep'+name+'.p'
    with open(title, "ab") as pickle_file:
        for t in test_val:
            try:
                log.write("start expl for "+str(t)+"\n")
                exp = explainer.explain_instance(t, samples=100, use_weights=True, metric=neuclidean)
                explainations.append((t, exp))
                pickle.dump(explainations, pickle_file)
                log.write("end expl for "+str(exp.rstr())+"\n")
                pickle.dump(explainations, pickle_file)
            except:
                continue


if __name__ == "__main__":
    main()

