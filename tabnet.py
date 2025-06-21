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

from src.preprocess import load_mutation_data, load_disease_data
import os
os.chdir(os.path.dirname(__file__))
def load_and_prepare_data():
    mutation_data = pd.read_csv("combined_mutation_matrix_full.csv")
    mutation_data = mutation_data.dropna(axis=1, how='any')
    disease_data = pd.read_csv("disease_ids.csv")

    mutation_data['disease_name'] = mutation_data['case'].map(disease_data.set_index('case_id')['tumor_code'])
    mutation_data = mutation_data.dropna(subset=['disease_name'])
    mutation_data = mutation_data.drop(columns=['case'])

    X = mutation_data.drop(columns=["disease_name"])
    X.dropna(axis=1, inplace=True)
    X = X.drop("worst_class_overall", axis=1, errors='ignore')
    X = X.loc[:, (X != 0).any(axis=0)]

    y = mutation_data['disease_name'].astype('category')
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    class_labels = le.classes_

    original_feature_names = X.columns.tolist()

    selector = SelectKBest(score_func=f_classif, k=2000)
    X_new = selector.fit_transform(X, y_encoded)
    selected_feature_names = [original_feature_names[i] for i in selector.get_support(indices=True)]

    return X_new, y_encoded, selected_feature_names, class_labels

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
    return {
        "model": clf,
        "X_test_selected": X_test,
        "y_test": y_test,
        "selected_feature_names": feature_names,
        "grid_search": None
    }

def print_metrics(model, X_test_selected, y_test, grid_search):
    y_pred = model.predict(X_test_selected)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

def plot_confusion_matrix(model, X_test_selected, y_test, class_labels, outdir="visualizations"):
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    y_pred = model.predict(X_test_selected)
    plt.figure(figsize=(6, 5))
    short_labels = ['ALL', 'AML']
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred,
        display_labels=short_labels,
        cmap='Blues'
    )
    plt.title("Confusion Matrix - TabNet")
    plt.savefig(f"{outdir}/tabnet_confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_feature_importance(model, feature_names, outdir="visualizations", top_n=15):
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    importances = model.feature_importances_
    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    print("\nTop features:")
    print(feat_imp.head(10))
    plt.figure(figsize=(10, 6))
    sns.barplot(x=feat_imp.head(top_n).values, y=feat_imp.head(top_n).index, palette='plasma')
    plt.title(f"Top {top_n} Feature Importances - TabNet")
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(f"{outdir}/tabnet_feature_importance_top20.png", dpi=300)
    plt.close()


def plot_roc_curve(model, X_test_selected, y_test, class_labels, outdir="visualizations"):
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    if len(class_labels) == 2:
        y_prob = model.predict_proba(X_test_selected)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_score = roc_auc_score(y_test, y_prob)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_score:.2f})", color='darkorange')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.title("ROC Curve - TabNet")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(f"{outdir}/tabnet_roc_curve.png", dpi=300)
        plt.close()

if __name__ == "__main__":
    X, y, feature_names, class_labels = load_and_prepare_data()
    model_info = train_model(X, y, feature_names)
    print_metrics(model_info["model"], model_info["X_test_selected"], model_info["y_test"], model_info["grid_search"])
    plot_confusion_matrix(model_info["model"], model_info["X_test_selected"], model_info["y_test"], class_labels)
    plot_feature_importance(model_info["model"], model_info["selected_feature_names"])
    plot_roc_curve(model_info["model"], model_info["X_test_selected"], model_info["y_test"], class_labels)