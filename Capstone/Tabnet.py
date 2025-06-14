from pytorch_tabnet.tab_model import TabNetClassifier
import matplotlib.pyplot as plt

import os
import torch
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay, roc_curve, roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.utils.class_weight import compute_sample_weight

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

    original_feature_names = X.columns.tolist()

    selector = SelectKBest(score_func=f_classif, k=2000)
    X_new = selector.fit_transform(X, y)
    selected_feature_names = [original_feature_names[i] for i in selector.get_support(indices=True)]

    return X_new, y, selected_feature_names

def train_model(X, y, feature_names, k=25):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=48)

    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    clf = TabNetClassifier(
        n_d=32, 
        n_a=32,
        n_steps=5,
        mask_type="entmax",
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
        eval_metric=["balanced_accuracy"],
        max_epochs=200,
        patience=10,
        batch_size=256, 
        virtual_batch_size=64,
        num_workers=0,
        drop_last=False,
        weights=sample_weights
    )

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.3f}")

    y_pred = clf.predict(X_test)
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred).plot()
    plt.title("Confusion Matrix - TabNet")
    plt.savefig("visualizations/tabnet_confusion_matrix.png")

    feat_importances = clf.feature_importances_
    feat_df = pd.DataFrame({'feature': feature_names, 'importance': feat_importances})
    feat_df = feat_df.sort_values(by='importance', ascending=False)

    plt.figure(figsize=(10, 8))
    sns.barplot(x='importance', y='feature', data=feat_df.head(20))
    plt.title('Tabnet: Top 20 Important Features')
    plt.xlabel('Belang')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig('visualizations/tabnet_feature_importance_top20.png', dpi=300, bbox_inches='tight')
    plt.close()
    y_proba = clf.predict_proba(X_test)
    n_classes = y_proba.shape[1]

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve((y_test == i).astype(int), y_proba[:, i])
        roc_auc[i] = roc_auc_score((y_test == i).astype(int), y_proba[:, i])

    plt.figure(figsize=(10, 8))
    for i in range(min(n_classes, 5)):
        plt.plot(fpr[i], tpr[i], label=f"Class {i} (AUC = {roc_auc[i]:.2f})")
    plt.plot([0, 1], [0, 1], 'k--', label='Random chance')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Multiclass ROC Curve - TabNet (Accuracy: {acc:.2f})")
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig("visualizations/tabnet_roc_curve.png", dpi=300, bbox_inches='tight')
    plt.close()



if __name__ == "__main__":
    X, y, names = load_and_prepare_data()
    model_info = train_model(X, y, names)

