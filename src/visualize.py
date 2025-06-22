import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from collections import Counter
from preprocess import load_mutation_data, load_disease_data
import os

plt.style.use("fivethirtyeight")
sns.set(font_scale=1.1)

df = load_mutation_data()
disease_data = load_disease_data()
df["disease_name"] = df["case"].map(disease_data.set_index("case_id")["tumor_code"])
df = df.dropna(subset=["disease_name"])
df = df.drop(columns=["case"])
df["group"] = df["disease_name"]

# Count the number of samples for each group
group_counts = df["group"].value_counts()
print(f"AML samples: {group_counts.get('Acute myeloid leukemia (AML)', 0)}")
print(f"ALL samples: {group_counts.get('Acute lymphoblastic leukemia (ALL)', 0)}")

if not os.path.exists("visualizations"):
    os.makedirs("visualizations")

# --- UNIFIED GENE-LEVEL (t_alt + n_alt) DATAFRAME ---
# This block will always create a DataFrame with columns as gene names and values as t_alt + n_alt per sample


def build_gene_alt_df(df, group_col="group"):
    # If MultiIndex columns (e.g., ('t_alt', 'TP53'))
    if isinstance(df.columns, pd.MultiIndex):
        gene_names = sorted(
            {col[1] for col in df.columns if isinstance(col, tuple) and len(col) == 2}
        )
        gene_alt_cols = {}
        for gene in gene_names:
            t_alt = (
                df[("t_alt", gene)].fillna(0) if ("t_alt", gene) in df.columns else 0
            )
            n_alt = (
                df[("n_alt", gene)].fillna(0) if ("n_alt", gene) in df.columns else 0
            )
            gene_alt_cols[gene] = t_alt + n_alt
        gene_alt_df = pd.DataFrame(gene_alt_cols, index=df.index)
    else:
        # Flat columns: look for t_alt_{GENE} and n_alt_{GENE}
        gene_names = sorted(
            set(
                col.replace("t_alt_", "").replace("n_alt_", "")
                for col in df.columns
                if col.startswith("t_alt_") or col.startswith("n_alt_")
            )
        )
        gene_alt_cols = {}
        for gene in gene_names:
            t_alt = (
                df.get(f"t_alt_{gene}", 0).fillna(0)
                if f"t_alt_{gene}" in df.columns
                else 0
            )
            n_alt = (
                df.get(f"n_alt_{gene}", 0).fillna(0)
                if f"n_alt_{gene}" in df.columns
                else 0
            )
            gene_alt_cols[gene] = t_alt + n_alt
        gene_alt_df = pd.DataFrame(gene_alt_cols, index=df.index)
    # Add group column for group-based plots
    gene_alt_df[group_col] = df[group_col].values
    return gene_alt_df


# Build the gene_alt_df at the top
try:
    gene_alt_df = build_gene_alt_df(df, group_col="group")
except Exception as e:
    print("Error building gene_alt_df:", e)
    gene_alt_df = None

# 1. Group Distribution
plt.figure(figsize=(8, 5))
sns.countplot(x="group", data=df)
plt.title("Distribution of Healthy vs AML vs ALL in the Dataset", fontsize=16)
plt.xlabel("Group", fontsize=14)
plt.ylabel("Count", fontsize=14)
plt.savefig("visualizations/group_distribution.png", dpi=300, bbox_inches="tight")
plt.close()

# 2. Chromosomal abnormality heatmap - mean values by group
chromosome_cols = df.columns[2:43]
aml_data = df[df["group"] == "Acute myeloid leukemia (AML)"][chromosome_cols]
all_data = df[df["group"] == "Acute lymphoblastic leukemia (ALL)"][chromosome_cols]

aml_means = aml_data.mean()
all_means = all_data.mean()

heatmap_data = pd.DataFrame({"AML": aml_means, "ALL": all_means})

colors = [(0, 0, 1), (1, 1, 1), (1, 0, 0)]
cmap = LinearSegmentedColormap.from_list("custom_diverging", colors, N=256)

plt.figure(figsize=(14, 8))
sns.heatmap(
    heatmap_data.T,
    cmap=cmap,
    center=0,
    vmin=-0.1,
    vmax=0.3,
    annot=True,
    fmt=".2f",
    linewidths=0.5,
    cbar_kws={"label": "Mean Abnormality Value"},
)
plt.title("Mean Chromosomal Abnormality by Group", fontsize=18)
plt.xlabel("Chromosome Arm", fontsize=14)
plt.ylabel("Group", fontsize=14)
plt.xticks(rotation=90)
plt.savefig("visualizations/abnormality_heatmap.png", dpi=300, bbox_inches="tight")
plt.close()


# 3. Frequency of abnormality patterns by group
def count_abnormalities(data):
    return (data != 0).sum().sort_values(ascending=False)


aml_abnormalities = count_abnormalities(aml_data)
all_abnormalities = count_abnormalities(all_data)

aml_percentages = (aml_abnormalities / len(aml_data)) * 100
all_percentages = (all_abnormalities / len(all_data)) * 100

top_aml = aml_percentages.nlargest(15)
top_all = all_percentages.nlargest(15)

fig, axes = plt.subplots(1, 2, figsize=(24, 8))

sns.barplot(x=top_aml.values, y=top_aml.index, ax=axes[0], color="#d62728")
axes[0].set_title("TOP 15 Chromosome Abnormalities in AML", fontsize=14)
axes[0].set_xlabel("Percentage of Samples (%)", fontsize=12)
axes[0].set_ylabel("", fontsize=12)

sns.barplot(x=top_all.values, y=top_all.index, ax=axes[1], color="#1f77b4")
axes[1].set_title("TOP 15 Chromosome Abnormalities in ALL", fontsize=14)
axes[1].set_xlabel("Percentage of Samples (%)", fontsize=12)
axes[1].set_ylabel("", fontsize=12)

plt.tight_layout()
plt.savefig("visualizations/top_abnormalities.png", dpi=300, bbox_inches="tight")
plt.close()

# 4. Correlation heatmap of chromosomal abnormalities
plt.figure(figsize=(16, 14))
correlation_matrix = df[chromosome_cols].corr()
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

sns.heatmap(
    correlation_matrix,
    mask=mask,
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    annot=False,
    square=True,
    linewidths=0.5,
    cbar_kws={"label": "Correlation Coefficient"},
)
plt.title("Correlation Between Chromosomal Abnormalities", fontsize=18)
plt.tight_layout()
plt.savefig("visualizations/correlation_heatmap.png", dpi=300, bbox_inches="tight")
plt.close()


# 5. Normal vs Abnormal karyotypes by group
def classify_karyotype(row):
    if ((row[2:-2] != 0) & (~row[2:-2].isna())).any():
        return "Abnormal"
    else:
        return "Normal"


df["karyotype_status"] = df.apply(classify_karyotype, axis=1)

karyotype_by_group = (
    pd.crosstab(df["group"], df["karyotype_status"], normalize="index") * 100
)

plt.figure(figsize=(10, 6))
karyotype_by_group.plot(kind="bar", stacked=True, colormap="Set2")
plt.title("Normal vs Abnormal Karyotypes by Group", fontsize=16)
plt.xlabel("Group", fontsize=14)
plt.ylabel("Percentage", fontsize=14)
plt.xticks(rotation=0)
plt.legend(title="Karyotype Status")
plt.savefig("visualizations/karyotype_status.png", dpi=300, bbox_inches="tight")
plt.close()


# 6. Patterns of multiple abnormalities
def count_abnormalities_per_sample(row):
    return ((row != 0) & (~pd.isna(row))).sum()


df["abnormality_count"] = df[chromosome_cols].apply(
    count_abnormalities_per_sample, axis=1
)

plt.figure(figsize=(12, 7))
sns.histplot(
    data=df,
    x="abnormality_count",
    hue="group",
    bins=15,
    multiple="dodge",
    shrink=0.8,
    kde=True,
)
plt.title(
    "Distribution of Number of Chromosomal Abnormalities per Patient", fontsize=16
)
plt.xlabel("Number of Abnormalities", fontsize=14)
plt.ylabel("Count", fontsize=14)
plt.legend(title="Group")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.savefig(
    "visualizations/abnormality_count_distribution.png", dpi=300, bbox_inches="tight"
)
plt.close()

# Remove karyotype logic and plot
# Instead, plot mutation burden and top mutated genes


def plot_mutation_burden(df, gene_cols, save_path=None, ax=None):
    mutation_burden = (df[gene_cols] > 0).sum(axis=1)
    if ax is None:
        plt.figure(figsize=(10, 6))
        sns.histplot(mutation_burden, bins=20, kde=True, color="purple")
        plt.title("Distribution of Mutated Genes per Sample", fontsize=16)
        plt.xlabel("Number of Mutated Genes", fontsize=14)
        plt.ylabel("Number of Samples", fontsize=14)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
    else:
        sns.histplot(mutation_burden, bins=20, kde=True, color="purple", ax=ax)
        ax.set_title("Mutated Genes per Sample")
        ax.set_xlabel("Number of Mutated Genes")
        ax.set_ylabel("Number of Samples")


def plot_top_mutated_genes(
    df, gene_cols, group_col, group_name, color, save_path=None, ax=None, top_n=10
):
    group_df = df[df[group_col] == group_name]
    gene_counts = (group_df[gene_cols] > 0).sum(axis=0).sort_values(ascending=False)
    top_genes = gene_counts.head(top_n)
    if ax is None:
        plt.figure(figsize=(10, 6))
        sns.barplot(x=top_genes.values, y=top_genes.index, color=color)
        plt.title(f"Top {top_n} Mutated Genes in {group_name}", fontsize=16)
        plt.xlabel("Number of Samples", fontsize=14)
        plt.ylabel("Gene", fontsize=14)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
    else:
        sns.barplot(x=top_genes.values, y=top_genes.index, color=color, ax=ax)
        ax.set_title(f"Top {top_n} Mutated Genes in {group_name}")
        ax.set_xlabel("Number of Samples")
        ax.set_ylabel("Gene")


# Identify gene columns (exclude non-gene columns and keep only numeric columns)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
gene_cols = [
    col for col in numeric_cols if col not in ["abnormality_count", "mutation_burden"]
]

# Plot mutation burden
df["mutation_burden"] = (df[gene_cols] > 0).sum(axis=1)
plt.figure(figsize=(10, 6))
sns.histplot(
    df,
    x="mutation_burden",
    hue="group",
    bins=20,
    multiple="dodge",
    kde=True,
    palette="Set2",
)
plt.title("Distribution of Mutated Genes per Sample by Group", fontsize=16)
plt.xlabel("Number of Mutated Genes", fontsize=14)
plt.ylabel("Number of Samples", fontsize=14)
plt.legend(title="Group")
plt.savefig("visualizations/mutation_burden_by_group.png", dpi=300, bbox_inches="tight")
plt.close()

# Plot top mutated genes for each group
for group, color in zip(
    df["group"].unique(), ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e"]
):
    plot_top_mutated_genes(
        df,
        gene_cols,
        "group",
        group,
        color,
        save_path=f"visualizations/top_mutated_genes_{group.replace(' ', '_')}.png",
        top_n=10,
    )

# Utility functions for plotting


def plot_group_distribution(df, save_path=None, ax=None):
    sns.countplot(x="group", data=df, ax=ax, palette="Set2")
    if ax is None:
        plt.title("Distribution of Healthy vs AML vs ALL in the Dataset", fontsize=16)
        plt.xlabel("Group", fontsize=14)
        plt.ylabel("Count", fontsize=14)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
    else:
        ax.set_title("Group Distribution", fontsize=14)
        ax.set_xlabel("Group")
        ax.set_ylabel("Count")


def plot_abnormality_count_distribution(df, save_path=None, ax=None):
    if "abnormality_count" not in df.columns:
        return
    sns.histplot(
        data=df,
        x="abnormality_count",
        hue="group",
        bins=15,
        multiple="dodge",
        shrink=0.8,
        kde=True,
        ax=ax,
        palette="Set2",
    )
    if ax is None:
        plt.title(
            "Distribution of Number of Chromosomal Abnormalities per Patient",
            fontsize=16,
        )
        plt.xlabel("Number of Abnormalities", fontsize=14)
        plt.ylabel("Count", fontsize=14)
        plt.legend(title="Group")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
    else:
        ax.set_title("Abnormality Count Distribution")
        ax.set_xlabel("Number of Abnormalities")
        ax.set_ylabel("Count")
        ax.grid(axis="y", linestyle="--", alpha=0.7)


# Update summary visualization (remove karyotype, add mutation burden)
from matplotlib.gridspec import GridSpec

plt.figure(figsize=(20, 14))
gs = GridSpec(2, 2)

ax1 = plt.subplot(gs[0, 0])
plot_group_distribution(df, ax=ax1)

ax2 = plt.subplot(gs[0, 1])
plot_abnormality_count_distribution(df, ax=ax2)

ax3 = plt.subplot(gs[1, 0])
sns.histplot(
    df,
    x="mutation_burden",
    hue="group",
    bins=20,
    multiple="dodge",
    kde=True,
    palette="Set2",
    ax=ax3,
)
ax3.set_title("Mutated Genes per Sample by Group")
ax3.set_xlabel("Number of Mutated Genes")
ax3.set_ylabel("Number of Samples")

ax4 = plt.subplot(gs[1, 1])
# Show top mutated genes for the largest group
largest_group = df["group"].value_counts().idxmax()
plot_top_mutated_genes(
    df,
    gene_cols,
    "group",
    "Acute lymphoblastic leukemia (ALL)",
    "#d62728",
    ax=ax4,
    top_n=20,
)

plt.suptitle(
    "Dataset Summary: Mutation Burden and Top Mutated Genes",
    fontsize=22,
    fontweight="bold",
    y=1.02,
)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("visualizations/summary.png", dpi=300, bbox_inches="tight")
plt.close()

print(
    "All visualizations have been updated for Masked Somatic Mutation data. Karyotype plot removed!"
)

# --- USE gene_alt_df FOR ALL GENE-LEVEL VISUALIZATIONS ---
if gene_alt_df is not None:
    # Top mutated genes per group (by t_alt + n_alt)
    for group, color in zip(
        gene_alt_df["group"].unique(), ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e"]
    ):
        group_df = gene_alt_df[gene_alt_df["group"] == group]
        gene_sums = group_df.drop(columns="group").sum().sort_values(ascending=False)
        top_genes = gene_sums.head(15)
        plt.figure(figsize=(14, 6))
        sns.barplot(x=top_genes.values, y=top_genes.index, color=color)
        plt.title(f"Top 15 Genes by (t_alt + n_alt) in {group}", fontsize=16)
        plt.xlabel("Total (t_alt + n_alt)")
        plt.ylabel("Gene")
        plt.tight_layout()
        plt.savefig(
            f'visualizations/top_abnormalities_{group.replace(" ", "_")}_alt.png',
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
    # Correlation heatmap
    gene_corr = gene_alt_df.drop(columns="group").corr()
    plt.figure(figsize=(16, 14))
    mask = np.triu(np.ones_like(gene_corr, dtype=bool))
    sns.heatmap(
        gene_corr,
        mask=mask,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        annot=False,
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "Correlation Coefficient"},
    )
    plt.title("Correlation Between Genes (t_alt + n_alt)", fontsize=18)
    plt.tight_layout()
    plt.savefig(
        "visualizations/correlation_heatmap_genes_alt.png", dpi=300, bbox_inches="tight"
    )
    plt.close()
    # Mutation burden per sample
    df["mutation_burden"] = gene_alt_df.drop(columns="group").sum(axis=1)
    plt.figure(figsize=(12, 7))
    sns.histplot(
        data=df,
        x="mutation_burden",
        hue="group",
        bins=15,
        multiple="dodge",
        kde=False,
        palette="Set2",
    )
    plt.title("Distribution of (t_alt + n_alt) per Patient", fontsize=16)
    plt.xlabel("Total (t_alt + n_alt)")
    plt.ylabel("Count")
    plt.legend(title="Group")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.savefig(
        "visualizations/abnormality_count_distribution_genes_alt.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    # Top mutated genes overall
    gene_sums = gene_alt_df.drop(columns="group").sum().sort_values(ascending=False)
    top_genes_total = gene_sums.head(20)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_genes_total.values, y=top_genes_total.index, color="#1f77b4")
    plt.title("Top 10 Mutated Genes (t_alt + n_alt)", fontsize=16)
    plt.xlabel("Total (t_alt + n_alt)")
    plt.ylabel("Gene")
    plt.tight_layout()
    plt.savefig(
        "visualizations/top10_genes_by_total_alt.png", dpi=300, bbox_inches="tight"
    )
    plt.close()
