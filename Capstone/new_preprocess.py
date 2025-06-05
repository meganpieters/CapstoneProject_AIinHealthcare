import pandas as pd
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
from model import train_model, print_metrics, plot_confusion_matrix, plot_feature_importance, plot_roc_curve
from sklearn.preprocessing import OneHotEncoder


# TODO: These genes are good foor AML, but not for ALL. We need more genes for ALL
GENES = ["TP53", "RUNX1", "DNMT3A", "FLT3", "IDH1", "IDH2", "NRAS", "NPM1", "BCR-ABL1"]

def collect_maf_files(data_dir="data", extension=".maf.gz"):
    files = []
    for root, _, fns in os.walk(data_dir):
        for fn in fns:
            if fn.endswith(extension):
                files.append((os.path.join(root, fn), Path(root).name))
    return files

def build_long_df(files, genes):
    recs = []
    for fp, case in tqdm(files):
        df = pd.read_csv(fp, sep='\t', comment='#', compression='gzip',
                         usecols=['Hugo_Symbol','Variant_Classification',
                                  't_alt_count','n_alt_count'])
        df = df[df.Hugo_Symbol.isin(genes)]
        if df.empty: continue
        grp = df.groupby('Hugo_Symbol').agg(
            n_mut=('Hugo_Symbol','size'),
            t_alt=('t_alt_count','sum'),
            n_alt=('n_alt_count','sum'),
            # take the most severe variant per gene (optional)
            worst_class=('Variant_Classification',
                         lambda x: x.mode().iat[0])
        ).reset_index()
        grp['case'] = case
        recs.append(grp)
    return pd.concat(recs, ignore_index=True)

def get_all_genes(files):
    """
    Extracts and returns a sorted list of unique genes across all MAF files.
    """
    all_genes = set()
    for filepath, _ in tqdm(files, desc="Collecting unique genes"):
        try:
            df = pd.read_csv(filepath, sep='\t', comment='#', compression='gzip')
            if not df.empty:
                all_genes.update(df['Hugo_Symbol'].dropna().unique())
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
    return sorted(all_genes)

def make_feature_matrix(long_df, genes):
    feat_counts = long_df.pivot(index='case',
                                columns='Hugo_Symbol',
                                values=['n_mut','t_alt','n_alt'])\
                         .fillna(0)

    class_df = long_df.pivot(index='case',
                             columns='Hugo_Symbol',
                             values='worst_class')\
                      .fillna('None')
    one_hot_parts = []
    for gene in class_df.columns:
        ohe = OneHotEncoder(sparse_output=False, dtype=int, handle_unknown='ignore')
        reshaped = class_df[[gene]]
        ohe_array = ohe.fit_transform(reshaped)
        ohe_df = pd.DataFrame(ohe_array,
                              index=reshaped.index,
                              columns=[f"{gene}_{cat}" for cat in ohe.categories_[0]])
        one_hot_parts.append(ohe_df)

    class_ohe = pd.concat(one_hot_parts, axis=1)
    X = pd.concat([feat_counts, class_ohe], axis=1)

    return X

def load_mutation_data():
    if not os.path.exists("combined_mutation_matrix_test_2.csv"):
        raise FileNotFoundError("combined_mutation_matrix.csv does not exist. Please run the data processing script first.")
    df = pd.read_csv("combined_mutation_matrix_test_2.csv")
    df = df.dropna(axis=1, how='any')
    return df

def load_disease_data():
    if not os.path.exists("disease_ids.csv"):
        raise FileNotFoundError("disease_ids.csv does not exist. Please run the disease ID script first.")
    df = pd.read_csv("disease_ids.csv")
    return df

def visualize_mutation_data(df):
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.drop(columns=['case_id']).corr(), annot=True, fmt=".2f", cmap='coolwarm')
    plt.title("Mutation Correlation Heatmap")
    plt.show()

def visualize_disease_distribution(df):
    plt.figure(figsize=(10, 6))
    sns.countplot(y='disease_name', data=df, order=df['disease_name'].value_counts().index)
    plt.title("Disease Distribution")
    plt.xlabel("Count")
    plt.ylabel("Disease Name")
    plt.show()

def test_with_model():
    mutation_data = load_mutation_data()
    disease_data = load_disease_data()

    mutation_data['disease_name'] = mutation_data['case'].map(disease_data.set_index('case_id')['tumor_code'])
    mutation_data = mutation_data.dropna(subset=['disease_name'])
    mutation_data = mutation_data.drop(columns=['case'])
    X = mutation_data.drop(columns=["disease_name"])
    X.dropna(axis=1, inplace=True)
    y = mutation_data["disease_name"]
    model_info = train_model(X, y)
    print_metrics(model_info["model"], model_info["X_test_selected"], model_info["y_test"], model_info["grid_search"])
    plot_confusion_matrix(model_info["model"], model_info["X_test_selected"], model_info["y_test"])
    plot_feature_importance(model_info["model"], model_info["selected_feature_names"])
    plot_roc_curve(model_info["model"], model_info["X_test_selected"], model_info["y_test"])

if __name__ == "__main__":
    files = collect_maf_files("data")
    genes = get_all_genes(files)
    print(f"Found {len(genes)} unique genes across all MAF files.")
    print(f"Genes: {genes[:10]}...")  # Display first 10 genes for brevity
    long_df = build_long_df(files, genes)
    X = make_feature_matrix(long_df, genes)
    X.to_csv("combined_mutation_matrix_test_2.csv")