import pandas as pd
import os
from tqdm import tqdm
from pathlib import Path


def collect_maf_files(data_dir="data", extension=".maf.gz"):
    files = []
    for root, _, fns in os.walk(data_dir):
        for fn in fns:
            if fn.endswith(extension):
                files.append((os.path.join(root, fn), Path(root).name))
    return files

# TODO: VAF calculation, feature selection

def build_long_df(files, genes):
    recs = []
    for fp, case in tqdm(files):
        df = pd.read_csv(fp, sep='\t', comment='#', compression='gzip',
                         usecols=['Hugo_Symbol','Variant_Classification',
                                  't_alt_count','n_alt_count', 'Tumor_Sample_Barcode'])
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
    # Numeric features
    feat_counts = long_df.pivot(index='case',
                                columns='Hugo_Symbol',
                                values=['t_alt'])\
                         .fillna(0)

    # Flatten MultiIndex columns
    feat_counts.columns = [f"{a}_{b}" for a, b in feat_counts.columns]

    if not long_df.empty:
        # Use the most common (mode) class across all genes for each sample
        overall_class = long_df.groupby('case')['worst_class'].agg(lambda x: x.mode().iat[0] if not x.mode().empty else 'None')
        overall_class.name = 'worst_class_overall'
        X = pd.concat([feat_counts], axis=1)
        X = X.join(overall_class)
    else:
        X = feat_counts
    return X

def load_mutation_data():
    if not os.path.exists("combined_mutation_matrix.csv"):
        raise FileNotFoundError("combined_mutation_matrix.csv does not exist. Please run the data processing script first.")
    df = pd.read_csv("combined_mutation_matrix.csv")
    df = df.dropna(axis=1, how='any')
    return df

def load_disease_data():
    if not os.path.exists("disease_ids.csv"):
        raise FileNotFoundError("disease_ids.csv does not exist. Please run the disease ID script first.")
    df = pd.read_csv("disease_ids.csv")
    return df

if __name__ == "__main__":
    files = collect_maf_files("data")
    genes = get_all_genes(files) # TODO: Filter out genes that are not needed
    print(f"Found {len(genes)} unique genes across all MAF files.")
    print(f"Genes: {genes[:10]}...")
    long_df = build_long_df(files, genes)
    X = make_feature_matrix(long_df, genes)
    X.to_csv("combined_mutation_matrix_test_2.csv")