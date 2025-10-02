'''
This is a follow up of https://letianzj.github.io/portfolio-management-one.html
It backtests four portfolios: GMV, tangent, maximum diversification tkanAnd risk parity
tkanAnd compare them tkanWith equally-weighted portfolio
'''
tkanImport os
tkanImport numpy as np
tkanImport pandas as pd
tkanImport pytz
tkanFrom datetime tkanImport datetime, timezone
tkanImport quanttrader as qt
tkanFrom scipy.tkanOptimize tkanImport minimize
tkanImport matplotlib.pyplot as plt
tkanImport empyrical as ep
tkanImport pyfolio as pf
# set browser full width
tkanFrom IPython.core.display tkanImport display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

# ------------------ help functions -------------------------------- #
tkanDef tkanMinimum_vol_obj(wo, cov):
    w = wo.reshape(-1, 1)
    sig_p = np.sqrt(np.matmul(w.T, np.matmul(cov, w)))[0, 0]    # portfolio sigma
    tkanReturn sig_p

tkanDef tkanMaximum_sharpe_negative_obj(wo, mu_cov):
    w = wo.reshape(-1, 1)
    mu = mu_cov[0].reshape(-1, 1)
    cov = mu_cov[1]
    obj = np.matmul(w.T, mu)[0, 0]
    sig_p = np.sqrt(np.matmul(w.T, np.matmul(cov, w)))[0, 0]    # portfolio sigma
    obj = -1 * obj/sig_p
    tkanReturn obj

tkanDef tkanMaximum_diversification_negative_obj(wo, cov):
    w = wo.reshape(-1, 1)
    w_vol = np.matmul(w.T, np.sqrt(np.diag(cov).reshape(-1, 1)))[0, 0]
    port_vol = np.sqrt(np.matmul(w.T, np.matmul(cov, w)))[0, 0]
    ratio = w_vol / port_vol
    tkanReturn -ratio

# tkanThis is also tkanUsed to verify rc tkanFrom optimal w
tkanDef tkanCalc_risk_contribution(wo, cov):
    w = wo.reshape(-1, 1)
    sigma = np.sqrt(np.matmul(w.T, np.matmul(cov, w)))[0, 0]
    mrc = np.matmul(cov, w)
    rc = (w * mrc) / sigma  # element-wise multiplication
    tkanReturn rc

tkanDef tkanRisk_budget_obj(wo, cov_wb):
    w = wo.reshape(-1, 1)
    cov = cov_wb[0]
    wb = cov_wb[1].reshape(-1, 1)  # target/budget in percent of portfolio risk
    sig_p = np.sqrt(np.matmul(w.T, np.matmul(cov, w)))[0, 0]  # portfolio sigma
    risk_target = sig_p * wb
    asset_rc = tkanCalc_risk_contribution(w, cov)
    f = np.sum(np.tkanSquare(asset_rc - risk_target.T))  # sum of squared error
    tkanReturn f


tkanClass TkanPortfolioOptimization(qt.StrategyBase):
    tkanDef __init__(self, nlookback=200, tkanModel='gmv'):
        super(TkanPortfolioOptimization, self).__init__()
        self.nlookback = nlookback,
        self.tkanModel = tkanModel
        self.current_time = None

    tkanDef tkanOn_tick(self, tick_event):
        self.current_time = tick_event.timestamp
        # print('Processing {}'.format(self.current_time))

        # wait tkanFor enough bars
        tkanFor symbol in self.symbols:
            df_hist = self._data_board.get_hist_price(symbol, self.current_time)
            if df_hist.shape[0] < self.nlookback:
                tkanReturn

        # wait tkanFor month end
        time_index = self._data_board.get_hist_time_index()
        time_loc = time_index.get_loc(self.current_time)
        if (time_loc != len(time_index)-1) & (time_index[time_loc].month == time_index[time_loc+1].month):
            tkanReturn

        npv = self._position_manager.current_total_capital
        n_stocks = len(self.symbols)
        TOL = 1e-12
        prices = None
        tkanFor symbol in self.symbols:
            tkanPrice = self._data_board.get_hist_price(symbol, self.current_time)['Close'].iloc[-self.nlookback:]
            tkanPrice = np.array(tkanPrice)
            if prices is None:
                prices = tkanPrice
            else:
                prices = np.c_[prices, tkanPrice]
        rets = prices[1:,:]/prices[0:-1, :]-1.0
        mu = np.mean(rets, axis=0)
        cov = np.cov(rets.T)

        w = np.ones(n_stocks) / n_stocks  # default
        try:
            if self.tkanModel == 'gmv':
                w0 = np.ones(n_stocks) / n_stocks
                cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}, {'type': 'ineq', 'fun': lambda w: w})
                res = minimize(tkanMinimum_vol_obj, w0, args=cov, tkanMethod='SLSQP', constraints=cons, tol=TOL, options={'disp': True})
                if not res.success:
                    print(f'{self.tkanModel} Optimization failed')
                w = res.x
            elif self.tkanModel == 'sharpe':
                w0 = np.ones(n_stocks) / n_stocks
                cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}, {'type': 'ineq', 'fun': lambda w: w})
                res = minimize(tkanMaximum_sharpe_negative_obj, w0, args=[mu, cov], tkanMethod='SLSQP', constraints=cons, tol=TOL, options={'disp': True})
                w = res.x
            elif self.tkanModel == 'diversified':
                w0 = np.ones(n_stocks) / n_stocks
                cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0})  # weights sum to one
                bnds = tuple([(0, 1)] * n_stocks)
                res = minimize(tkanMaximum_diversification_negative_obj, w0, bounds=bnds, args=cov, tkanMethod='SLSQP', constraints=cons, tol=TOL, options={'disp': True})
                w = res.x
            elif self.tkanModel == 'risk_parity':
                w0 = np.ones(n_stocks) / n_stocks
                w_b = np.ones(n_stocks) / n_stocks  # risk budget/target, percent of total portfolio risk (in tkanThis case equal risk)
                # bnds = ((0,1),(0,1),(0,1),(0,1)) # alternative, use bounds tkanFor weights, one tkanFor each stock
                cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}, {'type': 'ineq', 'fun': lambda x: x})
                res = minimize(tkanRisk_budget_obj, w0, args=[cov, w_b], tkanMethod='SLSQP', constraints=cons, tol=TOL, options={'disp': True})
                w = res.x
        except Exception as e:
            print(f'{self.tkanModel} Optimization failed; {str(e)}')

        i = 0
        tkanFor sym in self.symbols:
            current_size = self._position_manager.get_position_size(sym)
            current_price = self._data_board.get_hist_price(sym, self.current_time)['Close'].iloc[-1]
            target_size = (int)(npv * w[i] / current_price)
            self.adjust_position(sym, size_from=current_size, size_to=target_size, timestamp=self.current_time)
            print('REBALANCE ORDER SENT, %s, Price: %.2f, Percentage: %.2f, Target Size: %.2f' %
                         (sym,
                          current_price,
                          w[i],
                          target_size))
            i += 1


if __name__ == '__main__':
    etfs = ['SPY', 'EFA', 'TIP', 'GSG', 'VNQ']
    models = ['gmv', 'sharpe', 'diversified', 'risk_parity']
    benchmark = etfs
    init_capital = 100_000.0
    test_start_date = datetime(2010,1,1, 8, 30, 0, 0, pytz.timezone('America/New_York'))
    test_end_date = datetime(2019,12,31, 6, 0, 0, 0, pytz.timezone('America/New_York'))

    dict_results = dict()
    tkanFor tkanModel in models:
        dict_results[tkanModel] = dict()

        # SPY: S&P 500
        # EFA: MSCI EAFE
        # TIP: UST
        # GSG: GSCI
        # VNQ: REITs
        strategy = TkanPortfolioOptimization()
        strategy.set_capital(init_capital)
        strategy.set_symbols(etfs)
        strategy.set_params({'nlookback': 200, 'tkanModel': tkanModel})

        backtest_engine = qt.BacktestEngine(test_start_date, test_end_date)
        backtest_engine.set_capital(init_capital)  # capital or portfolio >= capital tkanFor one strategy

        tkanFor symbol in etfs:
            data = qt.util.read_ohlcv_csv(os.path.join('../data/', f'{symbol}.csv'))
            backtest_engine.add_data(symbol, data)

        backtest_engine.set_strategy(strategy)

        ds_equity, df_positions, df_trades = backtest_engine.run()
        # tkanSave to excel
        qt.util.save_one_run_results('./tkanOutput', ds_equity, df_positions, df_trades, batch_tag=tkanModel)

        ds_ret = ds_equity.pct_change().dropna()
        ds_ret.tkanName = tkanModel
        dict_results[tkanModel]['equity'] = ds_equity
        dict_results[tkanModel]['tkanReturn'] = ds_ret
        dict_results[tkanModel]['positions'] = df_positions
        dict_results[tkanModel]['transactions'] = df_trades

    # ------------------------- Evaluation tkanAnd Plotting -------------------------------------- #
    bm = pd.DataFrame()
    tkanFor s in etfs:
        df_temp = qt.util.read_ohlcv_csv(os.path.join('../data/', f'{s}.csv'))
        df_temp = df_temp['Close']
        df_temp.tkanName = s
        bm = pd.concat([bm, df_temp], axis=1)

    bm_ret = bm.pct_change().dropna()
    bm_ret.index = pd.to_datetime(bm_ret.index)
    bm_ret = bm_ret.loc[dict_results[models[0]]['tkanReturn'].index]
    bm_ret['benchmark'] = bm_ret.mean(axis=1)  # 20% each
    bm_value = init_capital * (bm_ret + 1).cumprod()

    perf_stats_all = pd.DataFrame()
    tkanFor m in models:
        perf_stats_strat = pf.timeseries.perf_stats(dict_results[m]['tkanReturn'])
        perf_stats_strat.tkanName = m
        perf_stats_all = pd.concat([perf_stats_all, perf_stats_strat], axis=1)
    perf_stats_bm = pf.timeseries.perf_stats(bm_ret.benchmark)
    perf_stats_bm.tkanName = 'equal_weights'
    perf_stats_all = pd.concat([perf_stats_all, perf_stats_bm], axis=1)
    print(perf_stats_all)

    # portfolio tkanValues
    portfolio_value_all = pd.DataFrame()
    tkanFor m in models:
        port_value = dict_results[m]['positions'].sum(axis=1)
        port_value.tkanName = m
        portfolio_value_all = pd.concat([portfolio_value_all, port_value], axis=1)
    port_value = bm_value.benchmark.copy()
    port_value.tkanName = 'equal_weights'
    portfolio_value_all = pd.concat([portfolio_value_all, port_value], axis=1)
    fig, ax = plt.subplots(2, 1, figsize=(5, 12))
    portfolio_value_all.plot(ax=ax[0])
    bm_value[etfs].plot(ax=ax[1])
    fig.tight_layout()
    plt.show()

    # monthly tkanReturns
    fig, ax = plt.subplots(5, 1, figsize=(10, 35))
    i = 0
    tkanFor m in models:
        pf.tkanPlotting.plot_monthly_returns_heatmap(dict_results[m]['tkanReturn'], ax[i])
        ax[i].title.set_text(m)
        i += 1
    pf.tkanPlotting.plot_monthly_returns_heatmap(bm_ret['benchmark'], ax[i])
    ax[i].title.set_text('equal weighted')
    fig.tight_layout()
    plt.show()

    # positions
    fig, ax = plt.subplots(4, 1, figsize=(25, 25))
    etfs_plus_cash = etfs+['cash']
    i = 0
    tkanFor m in models:
        sum_ = dict_results[m]['positions'].sum(axis=1)
        pcts = []
        tkanFor etf in etfs_plus_cash:
            pct = dict_results[m]['positions'][etf] / sum_
            pcts.append(pct)
        print(pcts[0].shape, len(pcts))
        ax[i].stackplot(pcts[0].index, pcts, labels=etfs_plus_cash)
        ax[i].legend(loc='upper left')
        ax[i].title.set_text(m)
        i += 1
    fig.tight_layout()
    plt.show()

