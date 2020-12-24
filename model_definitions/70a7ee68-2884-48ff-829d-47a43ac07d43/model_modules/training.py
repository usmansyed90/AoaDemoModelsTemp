import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib
import seaborn as sns

def train(data_conf, model_conf, **kwargs):
    """Python train method called by AOA framework

    Parameters:
    data_conf (dict): The dataset metadata
    model_conf (dict): The model configuration to use

    Returns:
    None:No return

    """

    hyperparams = model_conf["hyperParameters"]

    # load data & engineer
    walmart_df = pd.read_csv(data_conf['data_set_1'])
    print(walmart_df.head())
    # features = 'sepallength,sepalwidth,petallength,petalwidth'.split(',')
    # X = iris_df.loc[:, features]
    # y = iris_df['class']

    # print("Starting training...")
    # # fit model to training data
    # knn = KNeighborsClassifier(n_neighbors=hyperparams['n_neighbors'])
    # knn.fit(X,y)
    # print("Finished training")

    # joblib.dump(knn, 'artifacts/output/iris_knn.joblib')
    # print("Saved trained model")