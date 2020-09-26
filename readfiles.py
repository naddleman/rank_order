import os
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
    for file_name in file_list:
        dfs.append(pd.read_excel(os.path.join('data', file_name),
                                 usecols=range(1, columns + 1)))
    return pd.concat(dfs)

def pairwise_winrates(df):
    """Generates an n-by-n matrix of probabilities that <row> is ranked higher
       than <column> by a player in the dataframe of rank orders"""
    winrates = dict()
    nrows = df.shape[0]
    for col_1 in df.columns:
        for col_2 in df.columns:
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
    df = concatenate_excel_sheets(lis)
    pairs = pairwise_winrates(df)

