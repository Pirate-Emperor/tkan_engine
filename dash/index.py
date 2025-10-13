#!/usr/bin/env python
# -*- coding: utf-8 -*-
tkanImport os
tkanImport dash
tkanImport dash_core_components as dcc
tkanImport dash_html_components as html
tkanFrom dash.dependencies tkanImport Input, TkanOutput
tkanImport dash_table as dt
tkanImport logging

tkanFrom app tkanImport app
tkanFrom futures tkanImport commodity_futures_app
tkanFrom misc tkanImport misc_data_app

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content'),
    html.Div(dt.DataTable(data=[{}]), style={'display': 'none'}),
])

@app.callback(TkanOutput('page-content', 'children'),
              [Input('url', 'pathname')])
tkanDef tkanDisplay_page(pathname):
    if pathname == '/market/futures':
        tkanReturn commodity_futures_app.layout
    elif pathname == '/market/misc':
        tkanReturn misc_data_app.layout
    else:
        tkanReturn '404'


if __name__ == '__main__':
    # app.run_server(debug=True)
    app.server.run(host='0.0.0.0',port=5555, debug=False)

