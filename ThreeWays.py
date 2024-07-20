# this module aims to calculate the correlation:
# person correlation,spearman correlation and kendall's t
# note:there is also maturity algorithm in pandas,numpy
# scipy,and statsmodels and so on,this module just write
# my mind when I read market risk measurement and management
###########################################################
import pandas as pd
import numpy as  np
from collections import Iterable
from itertools import combinations


# person correlation
def person_cor(x, y):
    '''this function  is uesed to calculate person correlation,
    it requires pandas:import pandas as pd

    Arguments:
        data {[list,array,series]} -- [python's list,
                pandas's series,numpy's array]
    '''
    if isinstance(x, Iterable):
        x = pd.Series(x)
    else:
        print('x is not a iterable')
        return
    if isinstance(y, Iterable):
        y = pd.Series(y)
    else:
        print('y is not a iterable')
        return
    xy = x * y
    xx = x * x
    yy = y * y
    e_x = x.mean()
    e_y = y.mean()
    e_xy = xy.mean()
    e_x_e_y = x.mean() * y.mean()
    e_xx = xx.mean()
    e_yy = yy.mean()
    return (e_xy - e_x_e_y) / ((e_xx - e_x ** 2) * (e_yy - e_y ** 2)) ** 0.5


def spearman_cor(x, y):
    '''this function  is uesed to calculate person correlation,
    it requires pandas:import pandas as pd

    Arguments:
        data {[list,array,series]} -- [python's list,
                pandas's series,numpy's array]
    '''

    if isinstance(x, Iterable):
        x = pd.Series(x)
    else:
        print('x is not a iterable')
        return
    if isinstance(y, Iterable):
        y = pd.Series(y)
    else:
        print('y is not a iterable')
        return
    df = pd.DataFrame()
    df['x'] = x
    df['y'] = y
    # sort value according x
    df = df.sort_values('x')
    # x rank from low to high
    df['x_rank'] = range(1, len(x) + 1)
    # sort values according y
    df = df.sort_values('y')
    # y rank from low to high
    df['y_rank'] = range(1, len(y) + 1)
    # diff square
    df['diff'] = df['x_rank'] - df['y_rank']
    df['diff_square'] = df['diff'] ** 2
    # num length
    n = len(df)
    diff_2_sum = df['diff_square'].sum()
    return 1 - (6 * diff_2_sum / (n * (n * n - 1)))


def kendall_t(x, y):
    '''this function  is uesed to calculate person correlation,
    it requires pandas:import pandas as pd

    Arguments:
        data {[list,array,series]} -- [python's list,
                pandas's series,numpy's array]
    '''
    x = [1, 2, 3]
    y = [2, 3, 4]
    if isinstance(x, Iterable):
        x = pd.Series(x)
    else:
        print('x is not a iterable')
        return
    if isinstance(y, Iterable):
        y = pd.Series(y)
    else:
        print('y is not a iterable')
        return
    df = pd.DataFrame()
    df['x'] = x
    df['y'] = y
    # sort value according x
    df = df.sort_values('x')
    # x rank from low to high
    df['x_rank'] = range(1, len(x) + 1)
    # sort values according y
    df = df.sort_values('y')
    # y rank from low to high
    df['y_rank'] = range(1, len(y) + 1)
    tuple_list = list(zip(df['x_rank'], df['y_rank']))
    # tuple_list=list(zip(df['x_rank'],df['y_rank']))
    com_list = list(combinations(tuple_list, 2))
    com_list = [[com[0][0], com[0][1], com[1][0], com[1][1]] for com in com_list]
    com_df = pd.DataFrame(com_list)
    com_df.columns = ['x0', 'y0', 'x1', 'y1']
    com_df['num'] = (com_df['x0'] - com_df['y0']) * (com_df['x1'] - com_df['y1'])
    con_num = len(com_df[com_df['num'] > 0])
    dis_num = len(com_df[com_df['num'] < 0])
    equal_num = len(com_df[com_df['num'] == 0])
    n = len(com_df)
    return 2 * (con_num - dis_num) / (n * (n - 1))