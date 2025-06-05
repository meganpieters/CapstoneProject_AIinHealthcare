import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from collections import Counter
import os

plt.style.use('fivethirtyeight')
sns.set(font_scale=1.1)

df = pd.read_csv('combined_results.csv')
df = df.dropna(axis=1, how='any')

# Replace -1 values with np.nan for better visualization
df_clean = df.copy()
for col in df.columns[2:]:
    df_clean[col] = df_clean[col].replace(-1, np.nan)

# Create a new column: 'group' = 'Healthy' or 'Leukemia'
df['group'] = df['desease_name'].apply(lambda x: 'Healthy' if x == 'Healthy' else 'Leukemia')
df_clean['group'] = df['desease_name'].apply(lambda x: 'Healthy' if x == 'Healthy' else 'Leukemia')

# Count the number of samples for each group
group_counts = df['group'].value_counts()
print(f"Healthy samples: {group_counts.get('Healthy', 0)}")
print(f"Leukemia samples: {group_counts.get('Leukemia', 0)}")

if not os.path.exists('visualizations'):
    os.makedirs('visualizations')

# 1. Group Distribution
plt.figure(figsize=(8, 5))
sns.countplot(x='group', data=df)
plt.title('Distribution of Healthy vs Leukemia in the Dataset', fontsize=16)
plt.xlabel('Group', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.savefig('visualizations/group_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Chromosomal abnormality heatmap - mean values by group
chromosome_cols = df.columns[2:43]
healthy_data = df[df['group'] == 'Healthy'][chromosome_cols]
leukemia_data = df[df['group'] == 'Leukemia'][chromosome_cols]

healthy_means = healthy_data.mean()
leukemia_means = leukemia_data.mean()

heatmap_data = pd.DataFrame({
    'Healthy': healthy_means,
    'Leukemia': leukemia_means
})

colors = [(0, 0, 1), (1, 1, 1), (1, 0, 0)]
cmap = LinearSegmentedColormap.from_list('custom_diverging', colors, N=256)

plt.figure(figsize=(14, 8))
sns.heatmap(heatmap_data.T, cmap=cmap, center=0, 
            vmin=-0.1, vmax=0.3, annot=True, fmt='.2f', 
            linewidths=0.5, cbar_kws={'label': 'Mean Abnormality Value'})
plt.title('Mean Chromosomal Abnormality by Group', fontsize=18)
plt.xlabel('Chromosome Arm', fontsize=14)
plt.ylabel('Group', fontsize=14)
plt.xticks(rotation=90)
plt.savefig('visualizations/abnormality_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Frequency of abnormality patterns by group
def count_abnormalities(data):
    return (data != 0).sum().sort_values(ascending=False)

healthy_abnormalities = count_abnormalities(healthy_data)
leukemia_abnormalities = count_abnormalities(leukemia_data)

healthy_percentages = (healthy_abnormalities / len(healthy_data)) * 100
leukemia_percentages = (leukemia_abnormalities / len(leukemia_data)) * 100

top_healthy = healthy_percentages.nlargest(15)
top_leukemia = leukemia_percentages.nlargest(15)

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

sns.barplot(x=top_healthy.values, y=top_healthy.index, ax=axes[0], color='#2ca02c')
axes[0].set_title('TOP 15 Chromosome Abnormalities in Healthy', fontsize=14)
axes[0].set_xlabel('Percentage of Samples (%)', fontsize=12)
axes[0].set_ylabel('Chromosome Arm', fontsize=12)

sns.barplot(x=top_leukemia.values, y=top_leukemia.index, ax=axes[1], color='#d62728')
axes[1].set_title('TOP 15 Chromosome Abnormalities in Leukemia', fontsize=14)
axes[1].set_xlabel('Percentage of Samples (%)', fontsize=12)
axes[1].set_ylabel('', fontsize=12)

plt.tight_layout()
plt.savefig('visualizations/top_abnormalities.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. Correlation heatmap of chromosomal abnormalities
plt.figure(figsize=(16, 14))
correlation_matrix = df_clean[chromosome_cols].corr()
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

sns.heatmap(correlation_matrix, mask=mask, cmap='coolwarm', vmin=-1, vmax=1, 
            annot=False, square=True, linewidths=0.5, cbar_kws={'label': 'Correlation Coefficient'})
plt.title('Correlation Between Chromosomal Abnormalities', fontsize=18)
plt.tight_layout()
plt.savefig('visualizations/correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# 5. Normal vs Abnormal karyotypes by group
def classify_karyotype(row):
    if ((row[2:-2] != 0) & (~row[2:-2].isna())).any():
        return 'Abnormal'
    else:
        return 'Normal'

df['karyotype_status'] = df.apply(classify_karyotype, axis=1)

karyotype_by_group = pd.crosstab(df['group'], df['karyotype_status'], normalize='index') * 100

plt.figure(figsize=(10, 6))
karyotype_by_group.plot(kind='bar', stacked=True, colormap='Set2')
plt.title('Normal vs Abnormal Karyotypes by Group', fontsize=16)
plt.xlabel('Group', fontsize=14)
plt.ylabel('Percentage', fontsize=14)
plt.xticks(rotation=0)
plt.legend(title='Karyotype Status')
plt.savefig('visualizations/karyotype_status.png', dpi=300, bbox_inches='tight')
plt.close()

# 6. Patterns of multiple abnormalities
def count_abnormalities_per_sample(row):
    return ((row != 0) & (~pd.isna(row))).sum()

df['abnormality_count'] = df[chromosome_cols].apply(count_abnormalities_per_sample, axis=1)

plt.figure(figsize=(12, 7))
sns.histplot(data=df, x='abnormality_count', hue='group', bins=15, multiple='dodge', shrink=0.8, kde=True)
plt.title('Distribution of Number of Chromosomal Abnormalities per Patient', fontsize=16)
plt.xlabel('Number of Abnormalities', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.legend(title='Group')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('visualizations/abnormality_count_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# 7. Compare specific chromosome abnormalities between groups
chromosomes_of_interest = ['9q', '13q', '8p', '8q', '17p']

comparison_data = []
for chrom in chromosomes_of_interest:
    healthy_pct = (healthy_data[chrom] != 0).mean() * 100
    leukemia_pct = (leukemia_data[chrom] != 0).mean() * 100
    comparison_data.append({'Chromosome': chrom, 'Healthy': healthy_pct, 'Leukemia': leukemia_pct})

comparison_df = pd.DataFrame(comparison_data)
comparison_df = pd.melt(comparison_df, id_vars=['Chromosome'], var_name='Group', value_name='Percentage')

plt.figure(figsize=(12, 7))
sns.barplot(x='Chromosome', y='Percentage', hue='Group', data=comparison_df)
plt.title('Comparison of Selected Chromosomal Abnormalities Between Healthy and Leukemia', fontsize=16)
plt.xlabel('Chromosome', fontsize=14)
plt.ylabel('Percentage of Samples (%)', fontsize=14)
plt.legend(title='Group')
plt.savefig('visualizations/selected_chromosomes_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# 8. Create a summary visualization
from matplotlib.gridspec import GridSpec

plt.figure(figsize=(18, 14))
gs = GridSpec(2, 2)

ax1 = plt.subplot(gs[0, 0])
sns.countplot(x='group', data=df, ax=ax1)
ax1.set_title('Group Distribution')
ax1.set_xlabel('Group')
ax1.set_ylabel('Count')

ax2 = plt.subplot(gs[0, 1])
sns.boxplot(x='group', y='abnormality_count', data=df, ax=ax2)
ax2.set_title('Abnormalities per Patient')
ax2.set_xlabel('Group')
ax2.set_ylabel('Number of Abnormalities')

top5_healthy = healthy_percentages.nlargest(5)
top5_leukemia = leukemia_percentages.nlargest(5)

ax3 = plt.subplot(gs[1, 0])
sns.barplot(x=top5_healthy.values, y=top5_healthy.index, ax=ax3, color='#2ca02c')
ax3.set_title('Top 5 Abnormalities in Healthy')
ax3.set_xlabel('Percentage (%)')

ax4 = plt.subplot(gs[1, 1])
sns.barplot(x=top5_leukemia.values, y=top5_leukemia.index, ax=ax4, color='#d62728')
ax4.set_title('Top 5 Abnormalities in Leukemia')
ax4.set_xlabel('Percentage (%)')

plt.tight_layout()
plt.savefig('visualizations/summary.png', dpi=300, bbox_inches='tight')
plt.close()

print("All visualizations have been saved to the 'visualizations' directory.")
