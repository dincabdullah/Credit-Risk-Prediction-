import pickle


def get_all_trained_models():

    ret_dict = dict()

    with open("/Users/dincabdullah/Downloads/datawarehouse 2/creditriskpredictionapp/static/trainedmodels/decision_tree_classifier_model.pkl", 'rb') as file:
        decision_tree = pickle.load(file)
        ret_dict["decision_tree"] = decision_tree


    with open("/Users/dincabdullah/Downloads/datawarehouse 2/creditriskpredictionapp/static/trainedmodels/knn_classifier_model.pkl", 'rb') as file:
        knn = pickle.load(file)
        ret_dict["knn"] = knn


    with open("/Users/dincabdullah/Downloads/datawarehouse 2/creditriskpredictionapp/static/trainedmodels/svm_classifier_model.pkl", 'rb') as file:
        svm = pickle.load(file)
        ret_dict["svm"] = svm
    
    
    return ret_dict