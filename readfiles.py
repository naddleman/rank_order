""".xls files are 8 cols by n ~= 100 rows.
    first col is just ['', 'RR', '', '', ...] and each other col is the ranking
    of a sin, with rows being individuals' rankings. Top row is the sins, 
    rows 1:n are values (ranks)

    .xlsx files (from mTurk?) have id info in the first col, top two rows
    represent question information, and then the rest of the rows are
    responses.
"""

import os
<<<<<<< HEAD
import pandas as pd
=======
import itertools
import numpy as np
import pandas as pd
from collections import defaultdict
from math import factorial
>>>>>>> 970f02bd3eeb38ea2d55b88ab9fc8deabc06415d
import rankings

# PC_100_rank_time.xls does not have a header identifying sins and has 8 cols
# PA_100_rank_time.xls, PH..., PJ..., PM..., PP... have an 8th col (Incest)
sheets_xlsx = [fn for fn in os.listdir() if fn.endswith('.xlsx')]

def valid_sin_xls_sheets():
    sheets_xls  = [fn for fn in os.listdir('./data/') if fn.endswith('.xls')]
    eight_sins = ['PA', 'PH', 'PJ', 'PM', 'PP', 'PC']
    only_7 = [sheet for sheet in sheets_xls if not sheet[:2] in eight_sins]
    return only_7

def concatenate_excel_sheets(file_list, columns=7):
    """Produces a single df with all the rank order data from many excel
       spreadsheets. The first column is assumed to be irrelevant"""
    dfs = []
    dfs_with_filename = []
    for file_name in file_list:
        excel_file = pd.read_excel(os.path.join('data', file_name),
                                    usecols=range(1, columns + 1))
        excel_file['filename'] = os.path.basename(file_name)
        dfs_with_filename.append(excel_file)
    stacked_fn = pd.concat(dfs_with_filename, ignore_index=True)
    return stacked_fn.drop('filename', axis=1), stacked_fn

def remove_faulty_rows(df, n):
    """Drop rows from df if the subject failed to enter unique values"""
    return df.drop(df.loc[df.iloc[:, :7].sum(axis=1) != sum(range(1, n + 1))].index)
    #return df.drop(df.loc[df.sum(axis=1) != sum(range(1, n + 1))].index)

def print_cluster_distributions(df_fn, label='filename'):
    """prints the distributions of rows with each unique label belonging to
       clusters"""
    labels = df_fn[label].unique()
    num_clusters = len(df_fn['cluster'].unique())
    for lab in labels:
        counts = df_fn.loc[df_fn[label] == lab]['cluster'].value_counts()
        print(lab)
        for i in range(num_clusters):
            print("Cluster", i, ":", counts[i])

#def rankings_to_file_dict(df):
#    """takes the dataframe with the 'filename' column and produces a map from
#       ranking (row) to filename"""
#    for row in rankings_to

# SciPy has kendall tau. Implement bubble sort distance instead?
#def kendall_tau(ordering_1, ordering_2):
#    """A naive implementation O(n^2) of Kendall's tau as a distance between
#       permutations (rankings. Each list must have the same elements"""
#    assert set(ordering_1) == set(ordering_2)
#    count = len(ordering_1)
#    denominator = factorial(count) // factorial(2) // factorial(count - 2)
#    numerator = 0
#    for i in range(2, 
"""
TODO: PCA, umap on vectors of distances?
rbf kernel pca?



"""

    
                
#def main():


if __name__ == '__main__':
    lis = valid_sin_xls_sheets()
    df, df_fn = concatenate_excel_sheets(lis)
    df = remove_faulty_rows(df, 7)
    df_fn = remove_faulty_rows(df_fn, 7)
    pairs = rankings.pairwise_winrates(df)
    assignments, meds = rankings.k_medoids(df)
    clusters = rankings.reverse_assignments(assignments)
    df_fn = rankings.insert_cluster_column(df_fn, clusters)
    print_cluster_distributions(df_fn)
