from src.preprocess import load_disease_data, build_long_df, collect_maf_files, make_feature_matrix
import pandas as pd
from src.modelxgb import load_and_prepare_data, train_model, print_metrics
from scipy.stats import chi2_contingency
from genes import GENES

def select_important_genes_feature_matrix(feature_matrix, disease_df, gene_prefix='t_alt_', top_n=50, verbose=True):
    if 'case' not in feature_matrix.columns:
        feature_matrix = feature_matrix.reset_index()
    disease_df = disease_df.copy()
    if 'case' not in disease_df.columns and 'case_id' in disease_df.columns:
        disease_df['case'] = disease_df['case_id']
    merged = feature_matrix.merge(disease_df, on='case')
    gene_scores = []
    for col in merged.columns:
        if not col.startswith(gene_prefix):
            continue
        gene = col[len(gene_prefix):]
        x = merged[col]
        if not set(x.unique()).issubset({0, 1}):
            x = (x > 0).astype(int)
        table = pd.crosstab(x, merged['tumor_code'])
        if table.shape[0] < 2 or table.shape[1] < 2:
            continue
        chi2, p, _, _ = chi2_contingency(table)
        gene_scores.append((gene, chi2, p))
    gene_scores.sort(key=lambda x: -x[1])
    selected_genes = [g[0] for g in gene_scores[:top_n]]
    if verbose:
        print("Top selected genes:")
        for gene, chi2, p in gene_scores[:top_n]:
            print(f"{gene}: chi2={chi2:.2f}, p={p:.2e}")
    return selected_genes

def main():
    mutation_data_full = pd.read_csv("combined_mutation_matrix_full.csv")
    mutation_data_full = mutation_data_full.dropna(axis=1, how='any')
    disease_data = load_disease_data()
    files = collect_maf_files()

    ns = [10, 50, 100, 200, 400, 500]
    results = []

    for top_n in ns:
        print(f"\n=== Feature selection: top {top_n} genes ===")
        selected_genes = select_important_genes_feature_matrix(mutation_data_full, disease_data, gene_prefix='t_alt_', top_n=top_n, verbose=True)
        long_df = build_long_df(files, selected_genes)
        mutation_data = make_feature_matrix(long_df, selected_genes)
        mutation_data = mutation_data.reset_index()

        X, y, class_labels = load_and_prepare_data(mutation_data, disease_data)
        model_info = train_model(X, y)

        print_metrics(model_info["model"], model_info["X_test_selected"], model_info["y_test"], model_info["grid_search"], return_dict=True)
        n_cases = len(mutation_data)
        print(f"Cases with feature selection: {n_cases} - total cases: {len(mutation_data_full)}")

if __name__ == "__main__":
    main()