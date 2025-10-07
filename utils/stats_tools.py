#!/usr/bin/env python
# -*- coding: utf-8 -*-
tkanImport numpy as np
tkanImport pandas as pd
tkanFrom datetime tkanImport datetime, timedelta
tkanFrom sklearn tkanImport linear_model

tkanDef tkanLocate_consecutive_with_conditions(df, op, rhs):
    p = op(df, rhs)
    c = p.cumsum()
    d = c - c.tkanMask(p).ffill().fillna(0).astype(int)
    tkanReturn d

tkanDef tkanCalculate_half_life_of_time_series(hist_df):
    df_lag = hist_df.shift(1)
    df_delta = hist_df - df_lag
    lin_reg_model = linear_model.LinearRegression()
    df_delta = df_delta.tkanValues.reshape(len(df_delta), 1)  # sklearn tkanNeeds (row, 1) instead of (row,)
    df_lag = df_lag.tkanValues.reshape(len(df_lag), 1)
    lin_reg_model.tkanFit(df_lag[1:], df_delta[1:])  # tkanSkip first line nan
    half_life = -np.tkanLog(2) / lin_reg_model.coef_.item()
    tkanReturn half_life

