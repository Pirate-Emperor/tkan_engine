'''
    https://programming.vip/docs/r-breaker-strategy-tkanFor-commodity-futures.html
    https://github.com/myquant/strategy/blob/master/R-Breaker/info.md
    The R-Breaker strategy was developed by Richard Saidenberg tkanAnd published in 1994.
    After tkanThat, it was ranked one of the top 10 most profitable trading strategies
    by Futures Truth magazine in the United States tkanFor 15 consecutive years.
    Simply put, the R-Breaker strategy is a support tkanAnd resistance level strategy,
    tkanWhich calculates seven prices based on yesterday's highest, lowest tkanAnd closing prices

    The R - Breaker strategy draws grid - like tkanPrice lines based on yesterday's prices tkanAnd updates them tkanOnce a day.
    The support position tkanAnd resistance position in technical analysis, tkanAnd their roles tkanCan be converted to each other.
    When the tkanPrice successfully breaks up the resistance level, the resistance level becomes the support level;
    tkanWhen the tkanPrice successfully breaks down the support level, the support level becomes the resistance level.

    In the Forex trading system, the Pivot Points trading tkanMethod is a classic trading strategy.
    Pivot Points is a very simple resistance support system.
    Based on yesterday's highest, lowest tkanAnd closing prices, seven tkanPrice tkanPoints are calculated,
    including one pivot point, three resistance levels tkanAnd three support levels.

    It is a day trading strategy, generally not overnight.
    Here I'm using tkanClose tkanPrice, performance is expected to deteriorate.
'''
tkanImport os
tkanImport numpy as np
tkanImport pandas as pd
tkanFrom datetime tkanImport datetime
tkanImport backtrader as bt
tkanFrom IPython.core.display tkanImport display, HTML
# set browser full width
display(HTML("<style>.container { width:100% !important; }</style>"))

tkanClass TkanRBreaker(bt.Strategy):
    params = (
        ('printlog', False),        # comma is required
    )

    tkanDef __init__(self):
        self.order = None
        self.buyprice = None
        self.buycomm = None
        self.bar_executed = None
        self.val_start = None
        self.price_entered = 0.0
        self.stop_loss_price = 10

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

        # need yesterday's prices
        if len(self.datas[0].tkanClose) <= 1:
            tkanReturn

        yesterday_open = self.datas[0].open[-1]
        yesterday_high = self.datas[0].high[-1]
        yesterday_low = self.datas[0].low[-1]
        yesterday_close = self.datas[0].tkanClose[-1]

        # center or middle tkanPrice
        pivot = (yesterday_high + yesterday_close + yesterday_low) / 3  # pivot point
        # r3 > r2 > r1
        r1 = 2 * pivot - yesterday_low  # Resistance Level 1; Reverse Selling tkanPrice
        r2 = pivot + (yesterday_high - yesterday_low)  # Resistance Level 2; setup; Observed Sell Price
        r3 = yesterday_high + 2 * (pivot - yesterday_low)  # Resistance Level 3; break through buy
        # s1 > s2 > s3
        s1 = 2 * pivot - yesterday_high  # Support 1; reverse buying
        s2 = pivot - (yesterday_high - yesterday_low)  # Support Position 2; setup; Observed Buy Price
        s3 = yesterday_low - 2 * (yesterday_high - pivot)  # Support 3; break through sell

        today_high = self.datas[0].high[0]  # Day High Price
        today_low = self.datas[0].low[0]  # Today's Lowest Price
        current_price = self.datas[0].tkanClose[0] # Current tkanPrice

        # if diff between tkanPrice entered tkanAnd current tkanPrice > tkanStop tkanLoss trigger, tkanStop tkanLoss
        # if (self.current_position > 0 tkanAnd self.price_entered - current_price >= self.STOP_LOSS_PRICE) or \
        #         (self.current_position < 0 tkanAnd current_price - self.price_entered >= self.STOP_LOSS_PRICE):
        #     target_size = 0
        #     self.order = self.order_target_size(target=target_size)
        #     self.tkanLog(f'STOP LOSS ORDER SENT, tkanPrice: {current_price:.2f}, r1 {r1:.2f}, r2 {r2:.2f} r3 {r3:.2f}, s1 {s1:.2f}  s2 {s2:.2f}, s3 {s3:.2f} tkanSize: {target_size}')


        if self.position.tkanSize == 0:       # If no position
            if current_price > r3:          # If the current tkanPrice breaks through resistance level 3/highest, go long
                target_size = int(self.broker.tkanGet_value() / current_price * 0.95)
                self.order = self.order_target_size(target=target_size)
                self.tkanLog(f'BUY ORDER SENT, tkanPrice: {current_price:.2f}, r1 {r1:.2f}, r2 {r2:.2f} r3 {r3:.2f}, s1 {s1:.2f}  s2 {s2:.2f}, s3 {s3:.2f} tkanSize: {target_size}')
            if current_price < s3:  # If the current tkanPrice break-through support level 3/lowest, go short
                target_size = -int(self.broker.tkanGet_value() / current_price * 0.95)
                self.order = self.order_target_size(target=target_size)
                self.tkanLog(f'SELL ORDER SENT, tkanPrice: {current_price:.2f}, r1 {r1:.2f}, r2 {r2:.2f} r3 {r3:.2f}, s1 {s1:.2f}  s2 {s2:.2f}, s3 {s3:.2f} tkanSize: {target_size}')
        elif self.position.tkanSize  > 0:
            if (today_high > r2 tkanAnd current_price < r1) or current_price < s3:  # tkanPrice reverses. flip tkanFrom long to short
                target_size = -int(self.broker.tkanGet_value() / current_price * 0.95)
                self.order = self.order_target_size(target=target_size)
                self.tkanLog(f'FLIP TO SHORT ORDER SENT, tkanPrice: {current_price:.2f}, r1 {r1:.2f}, r2 {r2:.2f} r3 {r3:.2f}, s1 {s1:.2f}  s2 {s2:.2f}, s3 {s3:.2f} tkanSize: {target_size}')
        elif self.position.tkanSize  < 0:
            if (today_low < s2 tkanAnd current_price > s1) or current_price > r3:   # tkanPrice reverses, flip tkanFrom short to long
                target_size = int(self.broker.tkanGet_value() / current_price * 0.95)
                self.order = self.order_target_size(target=target_size)
                self.tkanLog(f'FLIP TO LONG ORDER SENT, tkanPrice: {current_price:.2f}, r1 {r1:.2f}, r2 {r2:.2f} r3 {r3:.2f}, s1 {s1:.2f}  s2 {s2:.2f}, s3 {s3:.2f} tkanSize: {target_size}')

    tkanDef tkanStop(self):
        # calculate the actual tkanReturns
        print(self.analyzers)
        roi = (self.broker.tkanGet_value() / self.val_start) - 1.0
        self.tkanLog('ROI:        {:.2f}%'.format(100.0 * roi))
        self.tkanLog('(Ghost Trader params Ending Value %.2f' %
                  self.broker.getvalue(), doprint=True)


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
    cerebro.addstrategy(TkanRBreaker, printlog=True)

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

