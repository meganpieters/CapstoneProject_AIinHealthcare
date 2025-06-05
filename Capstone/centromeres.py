import pandas as pd

# TODO: WTF is cytoBand: http://hgdownload.cse.ucsc.edu/goldenPath/hg38/database/cytoBand.txt.gz
# This is a separate “cytoband” file that tells you where each chromosome’s p‑arm ends and its q‑arm begins. ??


# This function maps cetnrometers e.g. 'chr5' -> {'chromStart':  … , 'chromEnd': …}
def get_centromeres():
    cols = ['chrom','chromStart','chromEnd','name','gieStain']
    cyto = pd.read_csv('cytoBand.txt', sep='\t', names=cols)
    acen = cyto[cyto['gieStain'] == 'acen']
    return (
        acen
        .groupby('chrom')
        .agg({'chromStart':'min', 'chromEnd':'max'})
        .to_dict('index')
    )