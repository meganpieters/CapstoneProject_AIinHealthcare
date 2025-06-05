import pandas as pd

df = pd.read_csv('combined_mutation_matrix.csv')

# print all rows, where all values are 0 except 'case_id'
zero_rows = df[(df.drop(columns=['case_id']) == 0).all(axis=1)]
print("Rows where all values are 0:")
print(zero_rows)