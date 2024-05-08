import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

from neural_networks import train_neural_network, get_best_neural_network

def decision_tree_classifier(X_train, y_train, X_test, y_test, results_path):
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)

    fig, axs = plt.subplots()
    tree.plot_tree(clf, ax=axs, feature_names=X_train.columns)
    plt.savefig('Results/decision_tree.svg')

    y_pred = clf.predict(X_test)
    evaluate_classification(y_test, y_pred, "decision tree")


def random_forest_classifier(X_train, y_train, X_test, y_test, results_path):
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    evaluate_classification(y_test, y_pred, "random forest")


def neural_network(X_train, y_train, X_test, y_test, results_path, retrain=True):
    nn_path = os.path.join(results_path, 'nn_weights')

    # Train the network either if no checkpoints exist or if the 'retrain' option is set.
    if retrain or not os.path.exists(nn_path) or not os.listdir(nn_path):
        train_neural_network(X_train, y_train, nn_path)
    # Always evaluate the network performance.
    nn_model = get_best_neural_network(X_test, y_test, nn_path)
    y_pred = nn_model.predict(X_test).round()
    evaluate_classification(y_test, y_pred, 'sequential nn')


def evaluate_classification(y_true, y_pred, model_name):
    print("{} accuracy: {:.3f}".format(model_name, metrics.accuracy_score(y_true, y_pred)))


def ml_analysis():
    data_path = 'Data/customer_churn.csv'
    df = pd.read_csv(data_path)

    # Dimensionality reduction: remove irrelevant variables.
    df = df.drop(['Call Failure', 'Age', 'Age Group', 'Subscription Length'], axis=1)

    # Convert seconds to minutes:
    df["Seconds of Use"] = df["Seconds of Use"] / 60
    df.rename(columns={"Seconds of Use": "Minutes of Use"}, inplace=True)

    X = df.drop('Churn', axis=1)
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    decision_tree_classifier(X_train, y_train, X_test, y_test, "Results")
    random_forest_classifier(X_train, y_train, X_test, y_test, "Results")
    neural_network(X_train, y_train, X_test, y_test, "Results", True)
