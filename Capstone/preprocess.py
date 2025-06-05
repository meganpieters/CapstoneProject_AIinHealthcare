import os
from pathlib import Path
from enum import IntEnum
import pandas as pd
import numpy as np
from centromeres import get_centromeres
import re

# Constants
GAIN_THRESHOLD = 2.5
LOSS_THRESHOLD = 1.5
NOISE_STD = 0.4
ARM_SHIFT_STD = 0.2
OUTLIER_RATE = 0.01

centromeres = get_centromeres()

class CopyNumberStatus(IntEnum):
    LOSS = -1
    NEUTRAL = 0
    GAIN = 1

def arm_for_segment(seg):
    if not seg.chromosome.startswith('chr'):
        seg.chromosome = 'chr' + str(seg.chromosome)

    centStart, centEnd = centromeres[str(seg.chromosome)]['chromStart'], centromeres[str(seg.chromosome)]['chromEnd']
    if seg.start < centStart:
        return 'p'
    elif seg.start > centEnd:
        return 'q'
    else:
        return 'centromere_overlap'

def summarize_arms(segments, label):
    segments['arm'] = segments.apply(arm_for_segment, axis=1)
    segments['chrom'] = segments['chromosome'].str.replace('chr', '')
    segments['chrom_arm'] = segments['chrom'] + segments['arm']

    arm_summary = {'desease_name': label}
    for arm in segments[segments['arm'] != 'centromere_overlap']['chrom_arm'].unique():
        arm_data = segments[segments['chrom_arm'] == arm]
        if arm_data.empty:
            continue

        gains = (arm_data['copy_number'] > GAIN_THRESHOLD).sum()
        losses = (arm_data['copy_number'] < LOSS_THRESHOLD).sum()
        total = len(arm_data)

        pct_gain = gains / total * 100
        pct_loss = losses / total * 100

        if pct_gain > 25:
            status = int(CopyNumberStatus.GAIN)
        elif pct_loss > 25:
            status = int(CopyNumberStatus.LOSS)
        else:
            status = int(CopyNumberStatus.NEUTRAL)

        arm_summary[arm] = status

    return pd.DataFrame([arm_summary])

def process_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} does not exist.")

    sample_id = Path(path).stem.split('.')[1]
    segments = pd.read_csv(path, sep='\t').dropna(subset=['copy_number'])

    return summarize_arms(segments, label="Infected")

def process_directory(directory):
    results = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.tsv'):
                try:
                    print(f"Processing file: {file}")
                    result = process_file(os.path.join(root, file))
                    results.append(result)
                except Exception as e:
                    print(f"Error processing file {file}: {e}")
    return pd.concat(results) if results else pd.DataFrame()

def generate_healthy_data(freq_path, num_samples=1000, noise=True):
    df = pd.read_csv(freq_path, sep="\t", comment="#")
    df['reference_name'] = df['reference_name'].astype(str)
    def extract_chromosome(ref):
        match = re.search(r'(?:chr)?(\d+|X|Y|MT)$', str(ref))
        return f'chr{match.group(1)}' if match else None

    df['chromosome'] = df['reference_name'].apply(extract_chromosome)

    df['arm'] = df.apply(arm_for_segment, axis=1)
    df['chrom'] = df['reference_name']
    df['chrom_arm'] = df['chrom'] + df['arm']

    df['gain_prob'] = df['gain_frequency'] / 100
    df['loss_prob'] = df['loss_frequency'] / 100

    samples = []
    for _ in range(num_samples):
        gains = np.random.binomial(1, df['gain_prob'])
        losses = np.random.binomial(1, df['loss_prob'])

        copy_number = 2 + gains - losses

        if noise:
            noise_vector = np.random.normal(0, NOISE_STD, size=len(df))
            arm_biases = {arm: np.random.normal(0, ARM_SHIFT_STD) for arm in df['chrom_arm'].unique()}
            arm_bias = df['chrom_arm'].map(arm_biases)
            outliers = np.random.rand(len(df)) < OUTLIER_RATE
            noise_vector[outliers] += np.random.normal(0, 2.0, size=outliers.sum())
            copy_number = copy_number + noise_vector + arm_bias

        sample = pd.DataFrame({
            'chromosome': df['reference_name'],
            'start': df['start'],
            'end': df['end'],
            'copy_number': copy_number,
            'arm': df['arm'],
            'chrom_arm': df['chrom_arm']
        })

        summary = summarize_arms(sample, label="Healthy")
        samples.append(summary)

    return pd.concat(samples, ignore_index=True)

def preprocess():
    print("Processing infected data...")
    infected = process_directory("./gdc_download_20250512_193121.287247")
    print("Infected data processed.")

    print("Processing healthy data...")
    num_healthy_total = len(infected)
    num_clean = int(num_healthy_total * 0.02) # 2% clean
    num_noisy = num_healthy_total - num_clean # remaining noisy
    print(f"Generating {num_clean} clean and {num_noisy} noisy samples.")

    healthy_clean = generate_healthy_data("./healthy/frequencies.pgxfreq", num_samples=num_clean, noise=False)
    healthy_noisy = generate_healthy_data("./healthy/frequencies.pgxfreq", num_samples=num_noisy, noise=True)
    healthy = pd.concat([healthy_clean, healthy_noisy], ignore_index=True)
    print("Healthy data processed.")

    combined = pd.concat([infected, healthy], ignore_index=True)
    combined = combined.dropna(axis=1, how='any')
    combined.to_csv("combined_results.csv", index=False)

def preprocess_clean():
    infected = pd.read_csv("combined_results.csv")
    infected = infected[infected['desease_name'] == 'Infected']

    print("Processing healthy data...")
    num_healthy_total = len(infected)
    healthy = generate_healthy_data("./healthy/frequencies.pgxfreq", num_samples=num_healthy_total, noise=True)
    print("Healthy data processed.")

    combined = pd.concat([infected, healthy], ignore_index=True)
    combined = combined.dropna(axis=1, how='any')
    combined.to_csv("combined_results.csv", index=False)

if __name__ == "__main__":
    preprocess()