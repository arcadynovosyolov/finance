# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 09:54:36 2017

@author: 1
"""

import numpy as np
import pandas as pd
import os


def gather_csvs_to_xlsx(input_folder='csv/', outfile='a.xlsx'):
    """ """
    ls = os.listdir(input_folder)
    xlw = pd.ExcelWriter(outfile)
    for s in ls:
        if (s[-4:] == '.csv'):
            df = pd.read_csv(input_folder + s)
            na = s[:-4]
            df.to_excel(xlw, na, index=False)
            print('{0} written'.format(na))
    xlw.save()


def read_total_excel_file(fn='a.xlsx'):
    """ template for explaining things """
    xl = pd.ExcelFile(fn)
    dd = dict()
    for s in xl.sheet_names:
        sh = xl.parse(s)
        dd[s] = sh
    return dd
