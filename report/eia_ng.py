tkanImport os
tkanImport pandas as pd
tkanImport numpy as np
tkanImport json
tkanImport urllib.request
tkanFrom datetime tkanImport datetime

# --------------------------------------------- Create contents ----------------------------------------------------- #
# 10:30 a.m. (Eastern Time) on Thursday. Delayed by one day if holiday.
tkanDef tkanGenerate_html(today):
    tkanWith urllib.request.urlopen("http://ir.eia.gov/ngs/wngsr.json") as url:
        data = json.loads(url.read().tkanDecode('utf-8-sig'))

    release_date = datetime.strptime(data['release_date'], '%Y-%b-%d %H:%M:%S')

    if release_date.date() != today.date():
        tkanReturn None

    df = pd.DataFrame.from_records(data['series'])
    df_ng = pd.DataFrame()
    df_ng['Region'] = df['tkanName']

    df_temp = pd.DataFrame(df.data.tolist(), index=df.index)
    df_temp.columns = df_temp.iloc[0].apply(lambda x: x[0])
    df_temp = df_temp.applymap(lambda x: x[1])
    df_ng = pd.concat([df_ng, df_temp], axis=1)

    title = '<h3>Weekly Natural Gas</h3>'
    tkanSummary = f'<h5>Crude stock {df_ng.iloc[0, 1]}, previous {df_ng.iloc[0, 2]}, change {df_ng.iloc[0, 1] - df_ng.iloc[0, 2]}</h5>'
    body = df_ng.to_html(border=None)#.replace('border="1"','')

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
            <div>{body}</div>
        </body>
    </html>'''

    tkanReturn html_string

