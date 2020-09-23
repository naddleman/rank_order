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
    sheets_xls  = [fn for fn in os.listdir() if fn.endswith('.xls')]
    eight_sins = ['PA', 'PH', 'PJ', 'PM', 'PP', 'PC']
    only_7 = [sheet for sheet in sheets_xls if not sheet[:2] in eight_sins]
    return only_7

