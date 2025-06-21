import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay, roc_curve, roc_auc_score

from preprocess import load_mutation_data, load_disease_data


def load_and_prepare_data():
    mutation_data = load_mutation_data()
    disease_data = load_disease_data()

    mutation_data['disease_name'] = mutation_data['case'].map(disease_data.set_index('case_id')['tumor_code'])
    mutation_data = mutation_data.dropna(subset=['disease_name'])
    mutation_data = mutation_data.drop(columns=['case'])
    X = mutation_data.drop(columns=["disease_name"])
    X.dropna(axis=1, inplace=True)
    X = X.drop("worst_class_overall", axis=1, errors='ignore')

    y = mutation_data['disease_name'].astype('category')
    return X, y


def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)
    return {
        "model": rf,
        "X_test_selected": X_test,
        "y_test": y_test,
        "selected_feature_names": X_train.columns,
        "selector": None,
        "X_train_selected": X_train,
        "y_train": y_train
    }

def print_metrics(model, X_test_selected, y_test):
    y_pred = model.predict(X_test_selected)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

def plot_confusion_matrix(model, X_test_selected, y_test, outdir="visualizations"):
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    plt.figure(figsize=(6, 5))
    short_labels = ['ALL', 'AML']
    ConfusionMatrixDisplay.from_estimator(model, X_test_selected, y_test, display_labels=short_labels, cmap='Blues')
    plt.title("Confusion Matrix - Tuned Random Forest")
    plt.savefig(f"{outdir}/rf_confusion_matrix.png", dpi=300, bbox_inches='tight')
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
    plt.title(f"Top {top_n} Feature Importances - Random Forest")
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(f"{outdir}/rf_feature_importance.png", dpi=300)
    plt.close()

def plot_roc_curve(model, X_test_selected, y_test, outdir="visualizations"):
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    if len(model.classes_) == 2:
        y_prob = model.predict_proba(X_test_selected)[:, 1]
        y_test_binary = y_test.map({'Acute lymphoblastic leukemia (ALL)': 0, 'Acute myeloid leukemia (AML)': 1})
        fpr, tpr, _ = roc_curve(y_test_binary, y_prob)
        auc_score = roc_auc_score(y_test_binary, y_prob)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_score:.2f})", color='darkorange')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.title("ROC Curve - Tuned Random Forest")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(f"{outdir}/rf_roc_curve.png", dpi=300)
        plt.close()

if __name__ == "__main__":
    X, y = load_and_prepare_data()
    model_info = train_model(X, y)
    print_metrics(model_info["model"], model_info["X_test_selected"], model_info["y_test"])
    plot_confusion_matrix(model_info["model"], model_info["X_test_selected"], model_info["y_test"])
    plot_feature_importance(model_info["model"], model_info["selected_feature_names"])
    plot_roc_curve(model_info["model"], model_info["X_test_selected"], model_info["y_test"])