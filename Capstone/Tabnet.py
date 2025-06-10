from pytorch_tabnet.tab_model import TabNetClassifier
import matplotlib.pyplot as plt

import os
import torch
import pandas as pd
import numpy as np

from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from preprocess import load_mutation_data, load_disease_data
import os
os.chdir(os.path.dirname(__file__))
def load_and_prepare_data():
    mutation_data = load_mutation_data()
    disease_data = load_disease_data()

    mutation_data['disease_name'] = mutation_data['case'].map(disease_data.set_index('case_id')['tumor_code'])
    mutation_data = mutation_data.dropna(subset=['disease_name'])
    mutation_data = mutation_data.drop(columns=['case'])

    X = mutation_data.drop(columns=["disease_name"])
    X.dropna(axis=1, inplace=True)
    X = X.drop("worst_class_overall", axis=1, errors='ignore')
    X = X.loc[:, (X != 0).any(axis=0)]

    y = mutation_data['disease_name'].astype('category')
    le = LabelEncoder()
    y = le.fit_transform(y)
    return X, y

def train_model(X, y, k=25):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Converteer expliciet naar NumPy arrays
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    clf = TabNetClassifier(
        n_d=32, 
        n_a=32,
        n_steps=5,
        gamma=1.5,
        lambda_sparse=1e-4,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=1e-3),
        scheduler_params={"step_size": 10, "gamma": 0.9},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        verbose=1
    )

    clf.fit(
        X_train=X_train, y_train=y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        eval_name=["train", "test"],
        eval_metric=["accuracy"],
        max_epochs=200,
        patience=20,
        batch_size=1024, 
        virtual_batch_size=128,
        num_workers=0,
        drop_last=False
    )


    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.3f}")

    feat_importances = clf.feature_importances_
    plt.barh(range(len(feat_importances)), feat_importances)
    plt.xlabel("Belang")
    plt.ylabel("Feature index")
    plt.title("Feature importance van genen")
    plt.show()

if __name__ == "__main__":
    X, y = load_and_prepare_data()
    model_info = train_model(X, y)