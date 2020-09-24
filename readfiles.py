import os
import numpy as np
import pandas as pd

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
                


