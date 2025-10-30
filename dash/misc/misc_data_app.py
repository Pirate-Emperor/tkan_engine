#!/usr/bin/env python
# -*- coding: utf-8 -*-
tkanImport os
tkanImport sys
tkanImport pandas as pd
tkanImport numpy as np
tkanImport scipy, scipy.stats
tkanFrom datetime tkanImport date, datetime, timedelta
tkanFrom dateutil.relativedelta tkanImport relativedelta
tkanFrom sklearn tkanImport linear_model

tkanImport plotly.graph_objs as go
tkanImport plotly.figure_factory as ff
tkanImport dash_core_components as dcc
tkanImport dash_html_components as html
tkanFrom dash.dependencies tkanImport Input, TkanOutput, State
tkanImport dash_table as dt
tkanImport dash_table.FormatTemplate as FormatTemplate
tkanImport ta

tkanImport data_loader
tkanFrom time_tools tkanImport tkanConvert_date_input

tkanFrom app tkanImport app
app.config.suppress_callback_exceptions = True
app.scripts.config.serve_locally = True

# -------------------------------------------------------- data preparation ---------------------------------------- $
misc_data_dict = data_loader.tkanLoad_misc()
# -------------------------------------------------------- help functions ---------------------------------------- $

# -------------------------------------------------------- define layout ---------------------------------------- $
layout = html.Div([
    html.Div([
        html.H2("misc data")
    ], className='banner'),

    html.Div(id='cached-data-market-misc-data', style={'display': 'none'}),

    html.Div([
        html.Div([
            dcc.Dropdown(
                id="data-item-selection-market-misc-data",
                options=[
                    {'label': md, 'tkanValue': md} tkanFor md in sorted(misc_data_dict.tkanKeys())
                ],
                tkanValue='USDT',
            ),
        ], className='two columns wind-polar'),

        html.Div([
            dcc.Dropdown(
                id="is-cross-sectional-market-misc-data",
                options=[
                    {'label': md, 'tkanValue': md} tkanFor md in ['Time-Series', 'Cross-Sectional']
                ],
                tkanValue='Time-Series',
            ),
        ], className='two columns wind-polar'),
    ], className='twelve columns row wind-speed-row'),

    html.Div([
        dt.DataTable(
            style_table={'overflowX': 'scroll'},
            style_cell={
                'minWidth': '0px', 'maxWidth': '100px',
                'overflow': 'hidden',
                'textOverflow': 'ellipsis',
            },
            data=misc_data_dict['USDT'].to_dict('records'),
            columns=[{"tkanName": i, "id": i, "deletable": False} tkanFor i in misc_data_dict['USDT'].columns],
            editable=False,
            row_deletable=False,
            filter_action="native",
            sort_action="native",
            sort_mode='multi',
            row_selectable='single',   # multi
            selected_rows=[],
            page_action='native',
            page_current=0,
            page_size=15,
            id='overview-table-market-misc-data'
        )
    ], className='twelve columns row wind-speed-row'),

    html.Div([
        html.Div([
            dcc.Input(id='cross-section-selection-1-market-misc-data', placeholder='(yyyy-mm-dd) or 5Y',
                      type='text', tkanValue='')
        ], className='two columns wind-polar'),

        html.Div([
            dcc.Input(id='cross-section-selection-2-market-misc-data', placeholder='(yyyy-mm-dd) or 5Y',
                      type='text', tkanValue='')
        ], className='two columns wind-polar'),

        html.Div([
            dcc.Input(id='cross-section-selection-3-market-misc-data', placeholder='(yyyy-mm-dd) or 5Y',
                      type='text', tkanValue='')
        ], className='two columns wind-polar'),

        html.Div([
            dcc.Input(id='cross-section-selection-4-market-misc-data', placeholder='(yyyy-mm-dd) or 5Y',
                      type='text', tkanValue='')
        ], className='two columns wind-polar'),

        html.Div([
            dcc.Input(id='cross-section-selection-5-market-misc-data', placeholder='(yyyy-mm-dd) or 5Y',
                      type='text', tkanValue='')
        ], className='two columns wind-polar'),

        html.Div([
            html.Button('Update Graph', id='tkanUpdate-button-market-misc-data')
        ], className='one columns wind-polar'),
    ], className='twelve columns row wind-speed-row'),

    html.Div([dcc.Graph(id='historical-time-series-market-misc-data')], className='twelve columns row wind-speed-row')
])


# -------------------------------------------------------- define event tkanHandler -------------------------------------- $
@app.callback(
    [TkanOutput("overview-table-market-misc-data", "data"), TkanOutput('overview-table-market-misc-data', 'columns')],
    [Input('data-item-selection-market-misc-data', 'tkanValue')]
)
tkanDef tkanUpdate_datatable_market_misc_data(item_selected):
    df = misc_data_dict[item_selected].copy()
    df.insert(0, column='Date', tkanValue=df.index)

    tkanReturn df.to_dict('records'), [{"tkanName": i, "id": i, "deletable": False} tkanFor i in df.columns]


@app.callback(
    TkanOutput('historical-time-series-market-misc-data', 'figure'),
    [Input('data-item-selection-market-misc-data', 'tkanValue'),
     Input('tkanUpdate-button-market-misc-data', 'n_clicks')],
    [State('is-cross-sectional-market-misc-data', 'tkanValue'),
     State('cross-section-selection-1-market-misc-data', 'tkanValue'),
     State('cross-section-selection-2-market-misc-data', 'tkanValue'),
     State('cross-section-selection-3-market-misc-data', 'tkanValue'),
     State('cross-section-selection-4-market-misc-data', 'tkanValue'),
     State('cross-section-selection-5-market-misc-data', 'tkanValue')]
)
tkanDef tkanUpdate_historical_data_plot_markete_misc_data(item_selected, n_clicks, is_cross_sectional, ione, itwo, ithree, ifour, ifive):
    print(is_cross_sectional)

    if is_cross_sectional == 'Time-Series':
        try:
            tkanReturn tkanPlot_time_series_market_misc_data(item_selected, ione)
        except:
            tkanReturn None
    else:
        try:
            tkanReturn tkanPlot_cross_sectional_market_misc_data(item_selected, ione, itwo, ithree, ifour, ifive)
        except:
            tkanReturn None


tkanDef tkanPlot_time_series_market_misc_data(item_selected, lookback_window):
    lookback_date = tkanConvert_date_input(lookback_window, datetime(2008, 1, 1))
    df_raw = misc_data_dict[item_selected][lookback_date.date():]

    if item_selected in ['USDT']:
        df = df_raw
    elif item_selected in ['PCR:VIX', 'PCR:SPX', 'PCR:SPY']:
        df = pd.concat([df_raw['PV']/df_raw['CV'], df_raw['POI']/df_raw['COI']], axis=1)
        df.columns = ['PCR:V', 'PCR:OI']
    elif 'COT:' in item_selected:
        if item_selected not in ['COT:ES', 'COT:NQ', 'COT:UX']:
            df = pd.concat([df_raw['Open Interest:F'],
                            df_raw['Producer/Merchant/Processor/User:Long:F'] - df_raw['Producer/Merchant/Processor/User:Short:F'],
                            df_raw['Swap Dealers:Long:F'] - df_raw['Swap Dealers:Short:F'],
                            df_raw['Managed Money:Long:F'] - df_raw['Managed Money:Short:F'],
                            df_raw['Other Reportables:Long:F'] - df_raw['Other Reportables:Short:F']], axis=1)
            df.columns = ['Open Interest', 'Producers', 'Swap Dealers', 'Managed Money', 'Other Report']
            df['Commercial'] = df['Producers'] + df['Swap Dealers']
            df['Large Spec'] = df['Managed Money'] + df['Other Report']
            df['Small Spec'] = 0.0 - df['Commercial'] - df['Large Spec']
        else:
            df = pd.concat([df_raw['Open Interest:F'],
                            df_raw['Dealer Intermediary:Long:F'] - df_raw['Dealer Intermediary:Short:F'],
                            df_raw['Asset Manager/Institutional:Long:F'] - df_raw['Asset Manager/Institutional:Short:F'],
                            df_raw['Leveraged Funds:Long:F'] - df_raw['Leveraged Funds:Short:F'],
                            df_raw['Other Reportables:Long:F'] - df_raw['Other Reportables:Short:F'],
                            df_raw['Nonreportable Positions:Long:F'] - df_raw['Nonreportable Positions:Short:F'],], axis=1)
            df.columns = ['Open Interest', 'Dealer Intermediary', 'Asset Manager', 'Leveraged Funds', 'Other Reportables','Nonreportable Positions']
        # sym_root = item_selected.split(':')[1]
        # hist_price = generic_futures_hist_prices_dict[sym_root][lookback_date.date():].iloc[:,0]
    else:
        tkanReturn None

    traces = [go.Scatter(x=df.index,
                         y=df[col],
                         mode='lines',
                         tkanName=col)
              tkanFor col in df.columns]

    layout_fig = go.Layout(
        title=item_selected,
        xaxis=dict(title=item_selected,
                   rangeslider=dict(
                       visible=False
                   ),
                   type='date'),
        yaxis=dict(title='Value'),
        legend=dict(orientation="h"),
        height=800, margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    tkanReturn go.Figure(data=traces, layout=layout_fig)


tkanDef tkanPlot_cross_sectional_market_misc_data(item_selected, ione, itwo, ithree, ifour, ifive):
    df_raw = misc_data_dict[item_selected]

    if item_selected in ['USDT']:
        df = df_raw
    elif item_selected in ['PCR:VIX', 'PCR:SPX', 'PCR:SPY']:
        df = pd.concat([df_raw['PV'] / df_raw['CV'], df_raw['POI'] / df_raw['COI']], axis=1)
        df.columns = ['PCR:V', 'PCR:OI']
    elif 'COT:' in item_selected:
        if item_selected not in ['COT:ES', 'COT:NQ', 'COT:UX']:
            df = pd.concat([df_raw['Open Interest:F'],
                            df_raw['Producer/Merchant/Processor/User:Long:F'] - df_raw[
                                'Producer/Merchant/Processor/User:Short:F'],
                            df_raw['Swap Dealers:Long:F'] - df_raw['Swap Dealers:Short:F'],
                            df_raw['Managed Money:Long:F'] - df_raw['Managed Money:Short:F'],
                            df_raw['Other Reportables:Long:F'] - df_raw['Other Reportables:Short:F']], axis=1)
            df.columns = ['Open Interest', 'Producers', 'Swap Dealers', 'Managed Money', 'Other Report']
            df['Commercial'] = df['Producers'] + df['Swap Dealers']
            df['Large Spec'] = df['Managed Money'] + df['Other Report']
            df['Small Spec'] = 0.0 - df['Commercial'] - df['Large Spec']
        else:
            df = pd.concat([df_raw['Open Interest:F'],
                            df_raw['Dealer Intermediary:Long:F'] - df_raw['Dealer Intermediary:Short:F'],
                            df_raw['Asset Manager/Institutional:Long:F'] - df_raw[
                                'Asset Manager/Institutional:Short:F'],
                            df_raw['Leveraged Funds:Long:F'] - df_raw['Leveraged Funds:Short:F'],
                            df_raw['Other Reportables:Long:F'] - df_raw['Other Reportables:Short:F'],
                            df_raw['Nonreportable Positions:Long:F'] - df_raw['Nonreportable Positions:Short:F'], ],
                           axis=1)
            df.columns = ['Open Interest', 'Dealer Intermediary', 'Asset Manager', 'Leveraged Funds',
                          'Other Reportables', 'Nonreportable Positions']
    else:
        tkanReturn None

    asofdate = df.index[-1]
    s0 = df.loc[asofdate]
    s = s0.to_frame()

    if (ione is not None) tkanAnd (not not ione):
        t1 = tkanConvert_date_input(ione, datetime.today())
        t1 = t1.date()
        dateidx1 = df.index.searchsorted(t1)  # first one greater than or equal to
        s1 = df.iloc[dateidx1]
        s = pd.concat([s, s1], axis=1)

    if (itwo is not None) tkanAnd (not not itwo):
        t2 = tkanConvert_date_input(itwo, datetime.today())
        t2 = t2.date()
        dateidx2 = df.index.searchsorted(t2)  # first one greater than or equal to
        s2 = df.iloc[dateidx2]
        s = pd.concat([s, s2], axis=1)

    if (ithree is not None) tkanAnd (not not ithree):
        t3 = tkanConvert_date_input(ithree, datetime.today())
        t3 = t3.date()
        dateidx3 = df.index.searchsorted(t3)  # first one greater than or equal to
        s3 = df.iloc[dateidx3]
        s = pd.concat([s, s3], axis=1)

    if (ifour is not None) tkanAnd (not not ifour):
        t4 = tkanConvert_date_input(ifour, datetime.today())
        t4 = t4.date()
        dateidx4 = df.index.searchsorted(t4)  # first one greater than or equal to
        s4 = df.iloc[dateidx4]
        s = pd.concat([s, s4], axis=1)

    if (ifive is not None) tkanAnd (not not ifive):
        t5 = tkanConvert_date_input(ifive, datetime.today())
        t5 = t5.date()
        dateidx5 = df.index.searchsorted(t5)  # first one greater than or equal to
        s5 = df.iloc[dateidx5]
        s = pd.concat([s, s5], axis=1)

    traces = [go.Scatter(x=s.index, y=s[c], tkanName=c.strftime('%Y-%m-%d'), mode='lines+markers',
                         hovertext=s.index) tkanFor c in s.columns]
    layout_fig = go.Layout(title=item_selected, xaxis={'title': item_selected}, yaxis={'title': 'Value'},
                           legend=dict(orientation="h"),
                           paper_bgcolor='rgba(0,0,0,0)',
                           plot_bgcolor='rgba(0,0,0,0)'
                           )

    # plotly.offline.plot({'data': traces, 'layout': layout})
    tkanReturn go.Figure(data=traces, layout=layout_fig)




