import os
import itertools
import numpy as np
import pandas as pd
from collections import defaultdict
from math import factorial


""".xls files are 8 cols by n ~= 100 rows.
    first col is just ['', 'RR', '', '', ...] and each other col is the ranking
    of a sin, with rows being individuals' rankings. Top row is the sins, 
    rows 1:n are values (ranks)

    .xlsx files (from mTurk?) have id info in the first col, top two rows
    represent question information, and then the rest of the rows are
    responses.
"""

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

def pairwise_winrates(df, data_columns=7):
    """Generates an n-by-n matrix of probabilities that <row> is ranked higher
       than <column> by a player in the dataframe of rank orders"""
    winrates = dict()
    nrows = df.shape[0]
    for col_1 in df.columns[:data_columns]:
        for col_2 in df.columns[:data_columns]:
            if col_1 == col_2:
                winrates[(col_1, col_1)] = 0.5
            else:
                difference = df[col_1] - df[col_2]
                count_wins = (difference < 0).value_counts()[True]
                winrates[(col_1, col_2)] = count_wins / nrows
    return winrates

def digraph_from_winrate_pairs(wr_dict):
    """Takes a dict with pairs of items for keys and win rate as values
       and produces a directed graph dictionary with keys being items
       and values, lists of items "beaten by" said item."""
    digraph = defaultdict(list)
    items = set()
    for k, v in wr_dict.items():
        head, tail = k[0], k[1]
        items += head
        items += tail
        if v > 0.5:
            digraph[head].append(tail)
    return digraph

def detect_cycles(digraph):
    """Performs a DFS on a digraph to detect cycles, returning a list of all
       cycles. The input should be represented as a dict whose keys are nodes
       and values are lists of nodes"""
    pass

def sigma(x):
    """helper function for Elo calculation"""
    return 1 / (1 + np.exp(-x))

def pr_prob_i_beats_j(i_rating, j_rating):
    """Sigmoid of the scaled ratings difference to help update Elo."""
    alpha = np.log(10) / 400
    return sigma(alpha * (i_rating - j_rating))

def elo_gradient_descent(wr_dict, n=5000):
    """calculates Elo ratings for each item in a simulated tournament
       where each player plays against the others. Using probability of winning
       should be equivalent to drawing a random ranking."""
    items = set()
    elo_dict = {}
    for k in wr_dict.keys():
        items = items.union({k[0], k[1]})
        elo_dict[k[0]] = 1000
    items_tup = tuple(items)
    K = 40 # temporary constant updating rate
    for i in range(n):
        if i > n//2:
            K = 20
        if i > (3 * n//4):
            K = 10
        matched_pair = tuple(np.random.choice(items_tup, 2, replace=False))
        i_item, j_item = matched_pair
        i_rating, j_rating = elo_dict[i_item], elo_dict[j_item]
        estimated_pr_i_wins = pr_prob_i_beats_j(i_rating, j_rating)
        estimated_pr_j_wins = 1 - estimated_pr_i_wins
        i_outcome = int(np.random.random() < wr_dict[matched_pair])
        j_outcome = 1 - i_outcome
        elo_dict[i_item] = i_rating + K * (i_outcome - estimated_pr_i_wins)
        elo_dict[j_item] = j_rating + K * (j_outcome - estimated_pr_j_wins)
    print("items: ", items)
    print("Elo dict:", elo_dict)
    print(sum(elo_dict.values()))
    return elo_dict

def sample_permutations_from_df(df):
    """Takes two random rows from the dataframe and returns lists representing
       the participant's ranking, e.g., [3, 5, 1, 7, 2, 4, 6]."""
    df_rows = df.sample(2)
    return df_rows.iloc[0], df_rows.iloc[1]

def kt_distance(perm0, perm1, normalized=False):
    """Counts the number of pairwise disagreements between rankings.
       This implementation is probably not the best."""
    length = len(perm0)
    assert len(perm0) == len(perm1)
    rename = dict(zip(perm0, range(1, length + 1)))
    renamed_p1 = list(map(lambda x: rename[x], perm1))
    distance = 0
    for j in range(1, length):
        for i in range(j):
            if renamed_p1[i] > renamed_p1[j]:
                distance += 1
    if not normalized:
        return distance
    else:
        return distance / (length * (length - 1) / 2)

def generate_distances_dict(n, metric=kt_distance):
    """Produces a dictionary storing all pairwise distances between points in
       permutation space (kt_distance). Too slow!"""
    distance_dict = {}
    sequence = range(1, n + 1)
    permuted = itertools.permutations(sequence)
    for pair in itertools.combinations(permuted, 2):
        f, s = pair
        distance_dict[pair] = metric(f, s)
    return distance_dict

def data_distances(df, metric=kt_distance):
    """Produces a dictionary storing all pairwise distances between data points
       from a dataframe of numerical rankings: elements in (1, .., n)

       uses kt_distance only, for now."""
    distance_dict = {}
    data_pairs = itertools.combinations(df.values, 2)
    for f, s in data_pairs:
        distance_dict[(tuple(f), tuple(s))] = metric(f, s)
    return distance_dict

def asymmetric_distance(p1, p2, distance_dict):
    """Since distance dict has key (A, B) but not (B, A) we will look up
       the swapped pair. Return 0 if p1 == p2"""
    if np.all(p1 == p2):
        return 0
    try:
        return distance_dict[(p1, p2)]
    except KeyError:
        return distance_dict[(p2, p1)]

def k_medoids(df, medoid_count=2, metric=kt_distance, max_iter=1000):
    """k-medoids clustering joins points to the closest mediod (like a median,
       but it must be a datapoint, not just a point in the ambient space.

       This is the naive implementation:
           1. Select n data points as initial medoids
           2. assign each data point to the closest medoid
           3. Determine a new medoid for each cluster
           4. Repeat (from 2) until the medoids do not change"""
    distances = data_distances(df, metric=metric)
    maxd = max(distances.values())
    medoids = list(map(tuple,df.sample(medoid_count).values))
    loss = maxd * len(df)
    #assignment step
    for iteration in range(max_iter):
        assignments = defaultdict(list)
        old_medoids = medoids.copy()
        print("old medoids:", old_medoids)
        for permutation in map(tuple,df.values):
            ds = [asymmetric_distance(x, permutation, distances)
                    for x in medoids]
            for i, d in enumerate(ds):
                if d == min(ds):
                    assignments[i].append(permutation)
        # determine new medoids
        for cluster in range(medoid_count):
            min_so_far = sum([asymmetric_distance(medoids[cluster], p1, distances)
                               for p1 in assignments[cluster]])
            for i, p0 in enumerate(assignments[cluster]):
                total = sum([asymmetric_distance(p0, p1, distances)
                                for p1 in assignments[cluster]])
                if total < min_so_far:
                    min_so_far = total
                    medoids[cluster] = p0
        print("new medoids:", medoids)
        if np.all(old_medoids == medoids):
            print("converged in " + str(iteration + 1) + " iterations.")
            break
    return assignments, medoids

def reverse_assignments(assignments):
    """reverses a dict: {k1: [v1, v2, ..]
                         k2: [v3, v4, ..]} ->
                        {v1: k1, v2: k1, v3: k2, v4: k2 ...}"""
    keys = assignments.keys()
    out_dict = {}
    for key in keys:
        for val in assignments[key]:
            out_dict[val] = key
    return out_dict

def insert_cluster_column(df, cluster_dict):
    """cluster_dict should be a dict of tuples -> cluster number
       maps a row to it's associated cluster and adds that column to df."""
    numerical_rows = len(list(cluster_dict.keys())[0])
    df['tuple'] = df.iloc[:, :numerical_rows].apply(tuple, axis=1)
    df['cluster'] = df['tuple'].apply(lambda x: cluster_dict[x])
    return df

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

    
                
#def main():


if __name__ == '__main__':
    lis = valid_sin_xls_sheets()
    df, df_fn = concatenate_excel_sheets(lis)
    df = remove_faulty_rows(df, 7)
    df_fn = remove_faulty_rows(df_fn, 7)
    pairs = pairwise_winrates(df)
    assignments, meds = k_medoids(df)
    clusters = reverse_assignments(assignments)
    df_fn = insert_cluster_column(df_fn, clusters)


