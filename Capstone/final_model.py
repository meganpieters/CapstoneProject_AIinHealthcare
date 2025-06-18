import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay, roc_curve, roc_auc_score
from sklearn.model_selection import cross_val_score, StratifiedKFold


from preprocess import load_mutation_data, load_disease_data
from skopt import BayesSearchCV


def load_and_prepare_data(mutation_data, disease_data):
    mutation_data['disease_name'] = mutation_data['case'].map(disease_data.set_index('case_id')['tumor_code'])
    mutation_data = mutation_data.dropna(subset=['disease_name'])
    mutation_data = mutation_data.drop(columns=['case'])

    X = mutation_data.drop(columns=["disease_name"])
    X.dropna(axis=1, inplace=True)
    X = X.drop("worst_class_overall", axis=1, errors='ignore')

    y = mutation_data['disease_name'].astype('category')
    y_encoded = y.cat.codes
    class_labels = y.cat.categories.tolist()
    return X, y_encoded, class_labels


def train_model(X, y, k=25, use_bayesian=False):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5],
        'learning_rate': [0.01, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'min_child_weight': [1, 5],
        'gamma': [0, 0.2],
        'reg_alpha': [0, 0.1],
        'reg_lambda': [1, 2]
    }
    xgb = XGBClassifier(eval_metric='mlogloss', random_state=42)
    if use_bayesian:
        search_spaces = {
            'n_estimators': (100, 200),
            'max_depth': (3, 5),
            'learning_rate': (0.01, 0.1, 'log-uniform'),
            'subsample': (0.8, 1.0),
            'colsample_bytree': (0.8, 1.0),
            'min_child_weight': (1, 5),
            'gamma': (0, 0.2),
            'reg_alpha': (0, 0.1),
            'reg_lambda': (1, 2)
        }
        search = BayesSearchCV(xgb, search_spaces, cv=5, n_jobs=-1, scoring='accuracy', verbose=1, n_iter=20, random_state=42)
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        grid_search = search
    else:
        grid_search = GridSearchCV(xgb, param_grid, cv=5, n_jobs=-1, scoring='accuracy', verbose=1)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_

    return {
        "model": best_model,
        "X_test_selected": X_test,
        "y_test": y_test,
        "selected_feature_names": X_train.columns,
        "selector": None,
        "X_train_selected": X_train,
        "y_train": y_train,
        "grid_search": grid_search
    }


def print_metrics(model, X_test_selected, y_test, grid_search, return_dict=False):
    y_pred = model.predict(X_test_selected)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    print("Accuracy:", acc)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    if return_dict:
        return {
            'accuracy': acc,
            'precision': report['weighted avg']['precision'],
            'recall': report['weighted avg']['recall'],
            'f1': report['weighted avg']['f1-score']
        }


def plot_confusion_matrix(model, X_test_selected, y_test, class_labels, outdir="visualizations"):
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    plt.figure(figsize=(6, 5))
    ConfusionMatrixDisplay.from_estimator(
        model, X_test_selected, y_test,
        display_labels=class_labels,
        cmap='Blues'
    )
    plt.title("Confusion Matrix - XGBoost")
    plt.savefig(f"{outdir}/xgb_confusion_matrix.png", dpi=300, bbox_inches='tight')
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
    plt.title(f"Top {top_n} Feature Importances - XGBoost")
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(f"{outdir}/xgb_feature_importance.png", dpi=300)
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
        plt.title("ROC Curve - XGBoost")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(f"{outdir}/xgb_roc_curve.png", dpi=300)
        plt.close()


def cross_validate_model(model, X, y, k=5):
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    print(f"\n{k}-Fold Cross-Validation Accuracy Scores: {scores}")
    print(f"Mean Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")
    return scores


if __name__ == "__main__":
    X, y, class_labels = load_and_prepare_data(load_mutation_data(), load_disease_data())
    model_info = train_model(X, y, use_bayesian=True)

    print("Best parameters from GridSearchCV:", model_info["grid_search"].best_params_)
    print_metrics(model_info["model"], model_info["X_test_selected"], model_info["y_test"], model_info["grid_search"])
    plot_confusion_matrix(model_info["model"], model_info["X_test_selected"], model_info["y_test"], class_labels)
    plot_feature_importance(model_info["model"], model_info["selected_feature_names"])
    plot_roc_curve(model_info["model"], model_info["X_test_selected"], model_info["y_test"], class_labels)
    cross_validate_model(model_info["model"], X, y, k=5)
