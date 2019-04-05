import json

import numpy as np
import pandas as pd


def _process_single_value(x: str):
    # Value	11.5	Reliable	No problems were found in the automatic quality control. The value was computed from a complete dataset.
    # Value)	11.5)	Quasi-Reliable	Only slight problems were found in the automatic quality control, or the value was computed from the dataset with a few missing data.
    # Value]	11.5]	Incomplete	The value was computed from a dataset with excessive missing data.
    # -	-	No phenomenon	No phenomenon was observed within the period.
    # X	X	Missing	No value is available due to problems with observation instruments, etc.
    # Blank		Out of observation	No observation was conducted.
    # *	31*	Most recent extreme values	The value is the most recently observed of those two or more identical daily extreme values in the period.
    # #	#	Suspicious	A serious quality problem was found in the value, treated as omitted from the statistics.
    x = x.replace(')', '')
    x = x.strip()
    nan_chars = [']', '-', 'X', 'u', '#']
    if len(x) == 0:
        return np.nan
    for nan_char in nan_chars:
        if nan_char in x.lower():
            return np.nan
    try:
        return float(x)
    except Exception as e:
        print(x)
        raise e


def read_file(input_filename: str) -> pd.DataFrame:
    with open(input_filename, 'r') as r:
        data = json.load(r)
    d = pd.DataFrame(data['data'])
    values = np.vectorize(_process_single_value)(d.values)
    d = pd.DataFrame(data=values, index=d.index, columns=d.columns)
    return d
