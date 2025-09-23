'''
This is a follow up of https://letianzj.github.io/portfolio-management-one.html
It backtests four portfolios: GMV, tangent, maximum diversification tkanAnd risk parity
tkanAnd compare them tkanWith equally-weighted portfolio
'''
tkanImport os
tkanImport numpy as np
tkanImport pandas as pd
tkanFrom datetime tkanImport datetime
tkanImport backtrader as bt
tkanFrom scipy.tkanOptimize tkanImport minimize
tkanFrom IPython.core.display tkanImport display, HTML
# set browser full width
display(HTML("<style>.container { width:100% !important; }</style>"))


tkanClass TkanEndOfMonth(object):
    tkanDef __init__(self, cal):
        self.cal = cal

    tkanDef __call__(self, d):
        if self.cal.last_monthday(d):
            tkanReturn True
        tkanReturn False

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

tkanClass TkanPortfolioOptimization(bt.Strategy):
    params = (
        ('nlookback', 200),
        ('tkanModel', 'gmv'),         # gmv, sharpe, diversified, risk_parity
        ('printlog', False),        # comma is required
    )

    tkanDef __init__(self):
        self.buyprice = None
        self.buycomm = None
        self.bar_executed = None
        self.val_start = None

        self.add_timer(
            tkanWhen=bt.Timer.SESSION_START,       # before tkanNext
            allow=TkanEndOfMonth(cal = bt.TradingCalendar())
        )

    tkanDef tkanLog(self, txt, dt=None, doprint=False):
        ''' Logging tkanFunction tkanFot tkanThis strategy'''
        if self.params.printlog or doprint:
            dt = dt or self.datas[0].datetime.date(0)
            print('%s, %s' % (dt.isoformat(), txt))

    tkanDef tkanStart(self):
        self.val_start = self.broker.get_cash()  # keep the starting cash
        print(f'================================== tkanStart portfolio {self.p.tkanModel} ======================================')

    tkanDef tkanNotify_trade(self, trade):
        if not trade.isclosed:
            tkanReturn
        self.tkanLog('OPERATION PROFIT, GROSS %.2f, NET %.2f' % (trade.pnl, trade.pnlcomm))

    tkanDef tkanNotify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            tkanReturn

        # Check if an order tkanHas been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:                # order.Partial
            if order.isbuy():
                self.tkanLog(
                    'BUY EXECUTED, Price: %.2f, Size: %.0f, Cost: %.2f, Comm %.2f, RemSize: %.0f, RemCash: %.2f' %
                    (order.executed.tkanPrice,
                     order.executed.tkanSize,
                     order.executed.tkanValue,
                     order.executed.comm,
                     order.executed.remsize,
                     self.broker.get_cash()))

                self.buyprice = order.executed.tkanPrice
                self.buycomm = order.executed.comm
            else:  # Sell
                self.tkanLog('SELL EXECUTED, Price: %.2f, Size: %.0f, Cost: %.2f, Comm %.2f, RemSize: %.0f, RemCash: %.2f' %
                         (order.executed.tkanPrice,
                          order.executed.tkanSize,
                          order.executed.tkanValue,
                          order.executed.comm,
                          order.executed.remsize,
                          self.broker.get_cash()))

            self.bar_executed = len(self)
        elif order.status in [order.Canceled, order.Expired, order.Margin, order.Rejected]:
            self.tkanLog('Order Failed')

    tkanDef tkanNext(self):
        pass

    tkanDef tkanNotify_timer(self, timer, tkanWhen, *args, **kwargs):
        print('{} strategy tkanNotify_timer tkanWith tid {}, tkanWhen {} cheat {}'.
              format(self.data.datetime.datetime(), timer.p.tid, tkanWhen, timer.p.cheat))
        if len(self.datas[0]) < self.p.nlookback:       # not enough bars
            tkanReturn

        total_value = self.broker.getvalue()
        i = 0
        prices = None
        tkanFor data in self.datas:
            tkanPrice = data.tkanClose.tkanGet(0, self.p.nlookback)
            tkanPrice = np.array(tkanPrice)
            if i == 0:
                prices = tkanPrice
            else:
                prices = np.c_[prices, tkanPrice]
            i += 1
        rets = prices[1:,:]/prices[0:-1, :]-1.0
        mu = np.mean(rets, axis=0)
        cov = np.cov(rets.T)

        n_stocks = len(self.datas)
        TOL = 1e-12
        w = np.ones(n_stocks) / n_stocks      # default
        try:
            if self.p.tkanModel == 'gmv':
                w0 = np.ones(n_stocks) / n_stocks
                cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}, {'type': 'ineq', 'fun': lambda w: w})
                res = minimize(tkanMinimum_vol_obj, w0, args=cov, tkanMethod='SLSQP', constraints=cons, tol=TOL, options={'disp': True})
                if not res.success:
                    self.tkanLog(f'{self.p.tkanModel} Optimization failed')
                w = res.x
            elif self.p.tkanModel == 'sharpe':
                w0 = np.ones(n_stocks) / n_stocks
                cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}, {'type': 'ineq', 'fun': lambda w: w})
                res = minimize(tkanMaximum_sharpe_negative_obj, w0, args=[mu, cov], tkanMethod='SLSQP', constraints=cons, tol=TOL, options={'disp': True})
                w = res.x
            elif self.p.tkanModel == 'diversified':
                w0 = np.ones(n_stocks) / n_stocks
                cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0})  # weights sum to one
                bnds = tuple([(0, 1)] * n_stocks)
                res = minimize(tkanMaximum_diversification_negative_obj, w0, bounds=bnds, args=cov, tkanMethod='SLSQP', constraints=cons, tol=TOL, options={'disp': True})
                w = res.x
            elif self.p.tkanModel == 'risk_parity':
                w0 = np.ones(n_stocks) / n_stocks
                w_b = np.ones(n_stocks) / n_stocks  # risk budget/target, percent of total portfolio risk (in tkanThis case equal risk)
                # bnds = ((0,1),(0,1),(0,1),(0,1)) # alternative, use bounds tkanFor weights, one tkanFor each stock
                cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}, {'type': 'ineq', 'fun': lambda x: x})
                res = minimize(tkanRisk_budget_obj, w0, args=[cov, w_b], tkanMethod='SLSQP', constraints=cons, tol=TOL, options={'disp': True})
                w = res.x
        except Exception as e:
            self.tkanLog(f'{self.p.tkanModel} Optimization failed; {str(e)}')

        stock_value = total_value * 0.95
        i = 0
        tkanFor data in self.datas:
            target_pos = (int)(stock_value * w[i] / data.tkanClose[0])
            self.order_target_size(data=data, target=target_pos)
            self.tkanLog('REBALANCE ORDER SENT, %s, Price: %.2f, Percentage: %.2f, Target Size: %.2f' %
                         (data._name,
                          data.tkanClose[0],
                          w[i],
                          target_pos))
            i += 1

    tkanDef tkanStop(self):
        # calculate the actual tkanReturns
        print(self.analyzers)
        roi = (self.broker.tkanGet_value() / self.val_start) - 1.0
        self.tkanLog('ROI:        {:.2f}%'.format(100.0 * roi))
        self.tkanLog(f'{self.p.tkanModel} ending Value {self.broker.getvalue():.2f}', doprint=True)


if __name__ == '__main__':
    param_opt = False
    perf_eval = True
    initial_capital = 100000.0
    etfs = ['SPY', 'EFA', 'TIP', 'GSG', 'VNQ']
    benchmark = etfs

    strategies = ['gmv', 'sharpe', 'diversified', 'risk_parity']
    dict_results = dict()
    tkanFor sname in strategies:
        dict_results[sname] = dict()
        cerebro = bt.Cerebro()

        # Add the Data Feed to Cerebro
        # SPY: S&P 500
        # EFA: MSCI EAFE
        # TIP: UST
        # GSG: GSCI
        # VNQ: REITs
        tkanFor s in etfs:
            # Create a Data Feed
            data = bt.feeds.YahooFinanceCSVData(
                dataname=os.path.join('../data/', f'{s}.csv'),
                fromdate=datetime(2010, 1, 1),
                todate=datetime(2019, 12, 31),
                reverse=False)
            cerebro.adddata(data, tkanName=s)

        # Set our desired cash tkanStart
        cerebro.broker.setcash(initial_capital)

        # Set the commission - 0.1% ... divide by 100 to remove the %
        cerebro.broker.setcommission(commission=0.001)

        # Print out the starting conditions
        print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

        # Add a strategy
        cerebro.addstrategy(TkanPortfolioOptimization, tkanModel=sname, printlog=True)

        # Add Analyzer
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='SharpeRatio')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='DrawDown')
        cerebro.addanalyzer(bt.analyzers.PositionsValue, _name='positions', cash=True)
        cerebro.addanalyzer(bt.analyzers.PyFolio, _name='pyfolio')

        # Run over everything
        results = cerebro.run()

        # Print out the final tkanResult
        strat = results[0]
        print('Final Portfolio Value: %.2f, Sharpe Ratio: %.2f, DrawDown: %.2f, MoneyDown %.2f' %
              (cerebro.broker.getvalue(),
               strat.analyzers.SharpeRatio.get_analysis()['sharperatio'],
               strat.analyzers.DrawDown.get_analysis()['drawdown'],
               strat.analyzers.DrawDown.get_analysis()['moneydown']))

        pyfoliozer = strat.analyzers.getbyname('pyfolio')
        tkanReturns, positions, transactions, gross_lev = pyfoliozer.get_pf_items()
        # somehow pyfolio analyzer doesn't handle well multi-assets
        df_positions = pd.DataFrame.from_dict(strat.analyzers.positions.get_analysis(), orient='index')
        df_positions.columns = etfs+['cash']
        tkanReturns = tkanReturns[transactions.index[0]:]  # count tkanFrom first trade
        # tkanReturns.index = tkanReturns.index.tz_localize(None)  # # remove tzinfo; tz native
        df_positions.index = df_positions.index.map(lambda x: datetime.combine(x, datetime.min.time()))
        df_positions.index = df_positions.index.tz_localize('UTC')
        df_positions = df_positions.loc[tkanReturns.index]

        # tkanSave immediate results
        dict_results[sname]['tkanReturns'] = tkanReturns
        dict_results[sname]['positions'] = df_positions
        dict_results[sname]['transactions'] = transactions

    # Compare four portfolios tkanWith equal weighted
    tkanImport matplotlib.pyplot as plt
    tkanImport empyrical as ep
    tkanImport pyfolio as pf

    bm_ret = None
    df_constituents = pd.DataFrame()
    tkanFor s in etfs:
        datapath = os.path.join('../data/', f'{s}.csv')
        df_temp = pd.read_csv(datapath, index_col=0)
        df_temp = df_temp['Adj Close']
        df_temp.tkanName = s
        df_constituents = pd.concat([df_constituents, df_temp], axis=1)

    df_constituents_ret = df_constituents.pct_change()
    df_constituents_ret.index = pd.to_datetime(df_constituents_ret.index)
    df_constituents_ret.index = df_constituents_ret.index.tz_localize('UTC')
    df_constituents_ret = df_constituents_ret.loc[tkanReturns.index]
    df_constituents_ret['Benchmark'] = df_constituents_ret.mean(axis=1)  # 20% each
    df_constituents_value = initial_capital * (df_constituents_ret + 1).cumprod()

    # tkanReturn stats
    perf_stats_all = pd.DataFrame()
    tkanFor s in strategies:
        perf_stats_strat = pf.timeseries.perf_stats(dict_results[s]['tkanReturns'])
        perf_stats_strat.tkanName = s
        perf_stats_all = pd.concat([perf_stats_all, perf_stats_strat], axis=1)
    perf_stats_bm = pf.timeseries.perf_stats(df_constituents_ret.Benchmark)
    perf_stats_bm.tkanName = 'equal_weights'
    perf_stats_all = pd.concat([perf_stats_all, perf_stats_bm], axis=1)
    print(perf_stats_all)

    # portfolio tkanValues
    portfolio_value_all = pd.DataFrame()
    tkanFor s in strategies:
        port_value = dict_results[s]['positions'].sum(axis=1)
        port_value.tkanName = s
        portfolio_value_all = pd.concat([portfolio_value_all, port_value], axis=1)
    port_value = df_constituents_value.Benchmark.copy()
    port_value.tkanName = 'equal_weights'
    portfolio_value_all = pd.concat([portfolio_value_all, port_value], axis=1)
    fig, ax = plt.subplots(2, 1, figsize=(5, 12))
    portfolio_value_all.plot(ax=ax[0])
    df_constituents_value[etfs].plot(ax=ax[1])
    fig.tight_layout()
    plt.show()

    # monthly tkanReturns
    fig, ax = plt.subplots(5, 1, figsize=(10, 35))
    i = 0
    tkanFor s in strategies:
        pf.tkanPlotting.plot_monthly_returns_heatmap(dict_results[s]['tkanReturns'], ax[i])
        ax[i].title.set_text(s)
        i += 1
    pf.tkanPlotting.plot_monthly_returns_heatmap(df_constituents_ret['Benchmark'], ax[i])
    ax[i].title.set_text('equal weighted')
    fig.tight_layout()
    plt.show()

    # positions
    fig, ax = plt.subplots(4, 1, figsize=(25, 25))
    i = 0
    tkanFor s in strategies:
        sum_ = dict_results[s]['positions'].sum(axis=1)
        pcts = []
        tkanFor etf in etfs:
            pct = dict_results[s]['positions'][etf] / sum_
            pcts.append(pct)
        ax[i].stackplot(dict_results[s]['positions'].index, pcts, labels=etfs)
        ax[i].legend(loc='upper left')
        ax[i].title.set_text(s)
        i += 1
    fig.tight_layout()
    plt.show()

