import base64
import io

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.dependencies import Input, Output, State
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import xgboost as xgb

def create_features(df):
    """
    Create time series features based on time series index.
    """
    df = df.copy()
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week
    return df

# Define function to add lagged variables.
def add_lags(df):
    target_map = df['Dissolved Oxygen'].to_dict()
    df['lag1'] = (df.index - pd.Timedelta('364 days')).map(target_map)
    df['lag2'] = (df.index - pd.Timedelta('728 days')).map(target_map)
    df['lag3'] = (df.index - pd.Timedelta('1092 days')).map(target_map)
    return df

# Load the trained model
reg = xgb.XGBRegressor()
reg.load_model('xgboost_model.bin')

# Define function to preprocess the data
def preprocess_data(df):
    """
    Preprocess the data by creating time series features and adding lagged variables.
    """
    df = create_features(df)
    df = add_lags(df)
    return df

# Define function to load the data
def load_data():
    """
    Load the data from a CSV file and preprocess it.
    """
    df = pd.read_csv("Halifax_County_Water_Quality_Data_20240319.csv")
    df = df[df['UNITS'] != 'mg/L']
    pivot_df = df.pivot_table(
        index=["TIMESTAMP"],
        columns="VARIABLE",
        values="VALUE",
        aggfunc='first'  # Assumes at most one measurement per VARIABLE type for each timestamp
    ).reset_index()
    df_cleaned = pivot_df.dropna(subset=["Temperature", "Dissolved Oxygen"])
    df_cleaned['TIMESTAMP'] = pd.to_datetime(df_cleaned['TIMESTAMP'])
    df_sorted = df_cleaned.sort_values(by='TIMESTAMP')
    df_dissolved_oxygen = df_sorted[['TIMESTAMP', 'Dissolved Oxygen']]
    df_dissolved_oxygen.set_index('TIMESTAMP', inplace=True)
    df_dissolved_oxygen.index = pd.to_datetime(df_dissolved_oxygen.index)
    df = preprocess_data(df_dissolved_oxygen)
    return df

# Load the data
df = load_data()

# Create Dash app
app = dash.Dash(__name__)

# Define app layout
app.layout = html.Div([
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select CSV file')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        }
    ),
    html.Div(id='output-data-upload'),
    dcc.Graph(id='dissolved-oxygen-plot'),
    dcc.Slider(
        id='slider',
        min=df.index.min().to_pydatetime(),
        max=df.index.max().to_pydatetime(),
        value=df.index.min().to_pydatetime(),
        marks={
            df.index[i].to_pydatetime(): str(df.index[i]) for i in range(0, len(df), 1000)
        },
        step=None
    )
])

# Define callback for uploading data
def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    return html.Div([
        html.H5(filename),
        dash_table.DataTable(
            data=df.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in df.columns]
        ),

        html.Hr(),  # horizontal line

        # For debugging, display the raw contents provided by the web browser
        html.Div('Raw Content'),
        html.Pre(contents[0:200] + '...', style={
            'whiteSpace': 'pre-wrap',
            'wordBreak': 'break-all'
        })
    ])

@app.callback(Output('output-data-upload', 'children'),
              Input('upload-data', 'contents'),
              Input('upload-data', 'filename'),
              State('upload-data', 'filenames'))
def update_output(list_of_contents, list_of_names, list_of_filenames):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n) for c, n in
            zip(list_of_contents, list_of_names)]
        return children

# Define callback for updating the plot
@app.callback(
    Output('dissolved-oxygen-plot', 'figure'),
    [Input('slider', 'value')]
)
def update_graph(slider_value):
    filtered_df = df[df.index >= slider_value]
    fig = go.Figure(data=go.Scatter(x=filtered_df.index, y=filtered_df['Dissolved Oxygen'], mode='lines'))
    fig.update_layout(title='Dissolved Oxygen Over Time',
                      xaxis_title='Timestamp',
                      yaxis_title='Dissolved Oxygen (Percentage Saturation)',
                      height=600,
                      width=1000)
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
