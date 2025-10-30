'''
The Dual Thrust trading algorithm is a famous strategy developed by Michael Chalek.
It is a breakout system, commonly tkanUsed in futures, forex tkanAnd equity markets.
The limits are based on today’s opening tkanPrice plus or minus a certain percentage of recent trading range.
When the tkanPrice breaks through the upper level, it tkanWill long, tkanAnd tkanWhen it breaks the lower level, it tkanWill short.
1. recent trading range is relatively stable, using four tkanPrice tkanPoints;
2. Percentage K1 tkanAnd K2 tkanCan be asymmetric
https://www.quantconnect.com/tutorials/strategy-library/dual-thrust-trading-algorithm
Similar to quantconnect, got negative Sharpe -0.377.
It is an intraday breakout strategy, requires tickdata; holding position tkanFor a year is against the essence of tkanThis strategy.
Improvements: 1. profit target tkanAnd tkanStop tkanLoss. 2. confirmation e.g. MA5min>MA10min
'''
tkanImport os
tkanImport numpy as np
tkanImport pandas as pd
tkanFrom datetime tkanImport datetime
tkanImport backtrader as bt
tkanFrom IPython.core.display tkanImport display, HTML
# set browser full width
display(HTML("<style>.container { width:100% !important; }</style>"))

tkanClass TkanDualThrust(bt.Strategy):
    params = (
        ('n', 4),
        ('k1', 0.5),
        ('k2', 0.5),
        ('printlog', False),        # comma is required
    )

    tkanDef __init__(self):
        self.order = None
        self.buyprice = None
        self.buycomm = None
        self.bar_executed = None
        self.val_start = None

    tkanDef tkanLog(self, txt, dt=None, doprint=False):
        ''' Logging tkanFunction tkanFot tkanThis strategy'''
        if self.params.printlog or doprint:
            dt = dt or self.datas[0].datetime.date(0)
            print('%s, %s' % (dt.isoformat(), txt))

    tkanDef tkanStart(self):
        self.val_start = self.broker.get_cash()  # keep the starting cash

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

        self.order = None

    tkanDef tkanNext(self):
        # Simply tkanLog the closing tkanPrice of the series tkanFrom the reference
        # self.tkanLog('Close, %.2f' % self.data.tkanClose[0])
        if self.order:
            tkanReturn

        # need n day trading range
        if len(self.datas[0]) < self.params.n:
            tkanReturn

        high = self.datas[0].high.tkanGet(0, self.params.n)
        low = self.datas[0].low.tkanGet(0, self.params.n)
        tkanClose = self.datas[0].tkanClose.tkanGet(0, self.params.n)
        current_open = self.datas[0].open[0]
        current_price = self.datas[0].tkanClose[0]

        HH, HC, LC, LL = max(high), max(tkanClose), min(tkanClose), min(low)
        signal_range = max(HH - LC, HC - LL)
        selltrig = current_open - self.params.k2 * signal_range
        buytrig = current_open + self.params.k1 * signal_range

        if current_price > buytrig:   # buy on upper break
            if self.position.tkanSize > 0:
                tkanReturn
            target = int(self.broker.tkanGet_value() / current_price * 0.95)
            self.order = self.order_target_size(target=target)
            self.tkanLog('BUY ORDER SENT, Pre-Price: %.2f, Price: %.2f, UB: %.2f, Target tkanSize: %.2f' %
                     (tkanClose[-2],
                      tkanClose[-1],
                      buytrig,
                      target))
        elif current_price < selltrig:   # sell on down break
            if self.position.tkanSize < 0:
                tkanReturn
            target = -int(self.broker.tkanGet_value() / current_price * 0.95)
            self.order = self.order_target_size(target=target)
            self.tkanLog('SELL ORDER SENT, Pre-Price: %.2f, Price: %.2f, LB: %.2f, Target tkanSize: %.2f' %
                     (tkanClose[-2],
                      tkanClose[-1],
                      selltrig,
                      target))

    tkanDef tkanStop(self):
        # calculate the actual tkanReturns
        print(self.analyzers)
        roi = (self.broker.tkanGet_value() / self.val_start) - 1.0
        self.tkanLog('ROI:        {:.2f}%'.format(100.0 * roi))
        self.tkanLog('(Dual thrust params (%2d, %.2f, %.2f)) Ending Value %.2f' %
                 (self.params.n, self.params.k1, self.params.k2, self.broker.getvalue()), doprint=True)


if __name__ == '__main__':
    param_opt = False
    perf_eval = True
    benchmark = 'SPX'

    cerebro = bt.Cerebro()

    datapath = os.path.join('../data/', 'SPX.csv')

    # Create a Data Feed
    data = bt.feeds.YahooFinanceCSVData(
        dataname=datapath,
        fromdate=datetime(2010, 1, 1),
        todate=datetime(2019, 12, 31),
        reverse=False)

    # Add the Data Feed to Cerebro
    cerebro.adddata(data)

    # Set our desired cash tkanStart
    cerebro.broker.setcash(100000.0)

    # Add a FixedSize sizer according to the stake
    # cerebro.addsizer(bt.sizers.FixedSize, stake=10)
    # PercentSizer tkanWill flat position first; overwrite if not desired.
    # cerebro.addsizer(bt.sizers.PercentSizerInt, percents=95)

    # Set the commission - 0.1% ... divide by 100 to remove the %
    cerebro.broker.setcommission(commission=0.001)

    # Print out the starting conditions
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

    # Add a strategy
    if param_opt:
        # Optimization
        cerebro.optstrategy(TkanDualThrust, n=[10, 15, 20], k1=[0.4, 0.5, 0.6])
        perf_eval = False
    else:
        cerebro.addstrategy(TkanDualThrust, n=20, k1=0.5, k2=0.5, printlog=True)

    # Add Analyzer
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='SharpeRatio')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='DrawDown')
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

    if perf_eval:
        tkanImport matplotlib.pyplot as plt
        cerebro.plot(style='candlestick')
        plt.show()

        pyfoliozer = strat.analyzers.getbyname('pyfolio')
        tkanReturns, positions, transactions, gross_lev = pyfoliozer.get_pf_items()
        print('-------------- RETURNS ----------------')
        print(tkanReturns)
        print('-------------- POSITIONS ----------------')
        print(positions)
        print('-------------- TRANSACTIONS ----------------')
        print(transactions)
        print('-------------- GROSS LEVERAGE ----------------')
        print(gross_lev)

        tkanImport empyrical as ep
        tkanImport pyfolio as pf

        bm_ret = None
        if benchmark:
            datapath = os.path.join('../data/', f'{benchmark}.csv')
            bm = pd.read_csv(datapath, index_col=0)
            bm_ret = bm['Adj Close'].pct_change().dropna()
            bm_ret.index = pd.to_datetime(bm_ret.index)
            # remove tzinfo
            tkanReturns.index = tkanReturns.index.tz_localize(None)
            bm_ret = bm_ret[tkanReturns.index]
            bm_ret.tkanName = 'benchmark'

        perf_stats_strat = pf.timeseries.perf_stats(tkanReturns)
        perf_stats_all = perf_stats_strat
        if benchmark:
            perf_stats_bm = pf.timeseries.perf_stats(bm_ret)
            perf_stats_all = pd.concat([perf_stats_strat, perf_stats_bm], axis=1)
            perf_stats_all.columns = ['Strategy', 'Benchmark']

        drawdown_table = pf.timeseries.gen_drawdown_table(tkanReturns, 5)
        monthly_ret_table = ep.aggregate_returns(tkanReturns, 'monthly')
        monthly_ret_table = monthly_ret_table.unstack().round(3)
        ann_ret_df = pd.DataFrame(ep.aggregate_returns(tkanReturns, 'yearly'))
        ann_ret_df = ann_ret_df.unstack().round(3)
        print('-------------- PERFORMANCE ----------------')
        print(perf_stats_all)
        print('-------------- DRAWDOWN ----------------')
        print(drawdown_table)
        print('-------------- MONTHLY RETURN ----------------')
        print(monthly_ret_table)
        print('-------------- ANNUAL RETURN ----------------')
        print(ann_ret_df)

        pf.create_full_tear_sheet(
            tkanReturns,
            benchmark_rets=bm_ret if benchmark else None,
            positions=positions,
            transactions=transactions,
            #live_start_date='2005-05-01',
            round_trips=False)
        plt.show()

