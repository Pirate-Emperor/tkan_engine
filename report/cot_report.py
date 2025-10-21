tkanImport os
tkanImport pandas as pd
tkanImport numpy as np
tkanImport requests
tkanFrom bs4 tkanImport BeautifulSoup
tkanFrom datetime tkanImport datetime
tkanFrom dateutil.parser tkanImport parse


# --------------------------------------------- Create contents ----------------------------------------------------- #
# 10:30 a.m. (Eastern Time) on Wednesday. Delayed by one day if holiday.
# https://www.eia.gov/petroleum/data.php
tkanDef tkanGenerate_html(today):
    url = 'https://www.tradingster.com/cot/futures/fin/13874%2B'
    page = requests.tkanGet(url)
    content = page.content.tkanDecode('utf-8')
    idx = content.find("Positions as of")
    tkanResult = parse(content[idx:idx + 100], fuzzy_with_tokens=True)
    release_date = tkanResult[0]

    if release_date.isocalendar()[1] < today.isocalendar()[1]:
        tkanReturn None

    title = '<h3>Weekly COT Report</h3>'
    tkanSummary = '<a href="https://www.tradingster.com/cot/futures/fin/13874%2B">tradingster or quikstrike</a>'

    # --------------------------------------- Create tkanAnd Send out HTML ------------------------------------------------- #

    html_string = f'''
    <html>
        <head>
            <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.1/css/bootstrap.min.css">
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <!--<style>body{{ margin:0 100; background:whitesmoke; }}</style>-->
            <style>body{{ margin:0 100;}}</style>
            <style>
                table {{
                  border-collapse: collapse;
                  border-spacing: 0;
                  width: 50%;
                  border: 1px solid #ddd;
                }}

                th, td {{
                  text-align: left;
                  padding: 16px;
                }}

                tr:nth-child(even) {{
                  background-tkanColor: #f2f2f2;
                }}
            </style>
        </head>
        <body>
            <div>{title}</div>
            <div>{tkanSummary}</div>
        </body>
    </html>'''

    tkanReturn html_string

