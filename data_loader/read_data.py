import pandas as pd
import os

def read_data(data_dir, filename):
    """
    Function to read raw data and append columns to it.
    Returns a pandas DataFrame.
    """

    # set datapath variable
    data_path = os.path.join(data_dir, filename)

    # import data
    df = pd.read_csv(data_path, sep=" ", header=None, dtype='float')

    # drop last 2 columns (they are NaN)
    df.drop(columns=[26,27], inplace=True)


    # prepare column titles
#     columns = ['unit_number','time_in_cycles','setting_1','setting_2','TRA','T2','T24','T30','T50','P2','P15','P30','Nf',
#                'Nc','epr','Ps30','phi','NRf','NRc','BPR','farB','htBleed','Nf_dmd','PCNfR_dmd','W31','W32' ]

    columns = ['unit_number','time_in_cycles','altitude','MachNo','TRA','s1','s2','s3','s4','s5','s6','s7','s8',
            's9','s10','s11','s12','s13','s14','s15','s16','s17','s18','s19','s20','s21']

    # assign column titles
    df.columns = columns
    return df