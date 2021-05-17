
import dash
import dash_table as dst
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc

import base64
import numpy as np
import pandas as pd
import datetime as dt

from dash.dependencies import Input, Output

from src.util import my_plot, model_feature_importance

model_name = 'RandomForestRegressor'
dash_title = "Coal Trend Forecast Report 05-17"
report_title = "Coal Trend Forecast Report 2021-05-17"
model_image_filename = "src/model_architecture.png"
model_encoded_image = base64.b64encode(open(model_image_filename, "rb").read()).decode("ascii")

model_benchmark = pd.read_csv("src/model_benchmark.csv")
model_benchmark = model_benchmark.pivot(index='model',columns='test year',values='mothly_acc')

report_list = ["20210517"]

FONT_AWESOME = "https://use.fontawesome.com/releases/v5.7.2/css/all.css"
app = dash.Dash(
    __name__, 
    title = dash_title, 
    suppress_callback_exceptions = True, 
    external_stylesheets = [dbc.themes.BOOTSTRAP, FONT_AWESOME]
    )

app.layout = html.Div([
    html.H1([report_title]), 
    html.Br(), 
    html.H2(["Introduction"]), 
    dbc.Row([
        dbc.Col([
            html.Br(),
            dcc.Markdown('''
            - Model feature engineering: 
                - Filter out features with 1% missing values
                - Linear Interpolate
                - Select K Best Features
            
            - Basic Model: 
                - RandomForest: fit trend

            - Report: 
            '''), 
            dbc.Row([
                dbc.Col([
                    ], width = 1), 
                dbc.Col([
                    dcc.Dropdown(
                        id = "simple-dropdown", 
                        options = [{"label": i, "value": i} for i in report_list], 
                        value = report_list[0], 
                        clearable = False, 
                        style = {"width": "150px"}
                        )
                    ])
                ])
            ], width = 3.5), 
        dbc.Col([
            html.H4(["Model Structure"]), 
            html.Br(),
            html.Img(src = "data:image/png;base64,{}".format(model_encoded_image)), 
            ], width = 2.5), 
        dbc.Col([
            html.H4(["Feature Importance"]), 
            html.Br(), 
            dcc.Graph(id = "fig-importance")], width = 6), 
        ]), 
    html.Br(), 
    html.H2(["Forecast"]), 
    html.Br(),
    dbc.Row([
        dbc.Col([
            dcc.Graph(
                id = "fig-week"
                )
            ]), 
        dbc.Col([
            dcc.Graph(
                id = "fig-month"
                )
            ])
        ]),
    html.Br(),
    dbc.Row([
        dbc.Col([ 
            html.Br(), 
            html.H2(["Performance"]), 
            html.Br(), 
            dcc.Markdown('''
            Test the monthly accuracy of the model every year from 2016 to 2020.
            '''), 
            html.Br(), 
            dst.DataTable(
                id='table',
                columns=[{"name": i, "id": i} for i in model_benchmark.columns],
                data=model_benchmark.to_dict('records'))
            ]),
    ]),
    html.Br(),
    html.Br(),
    ], style = {"margin-left": "2rem", "margin-right": "2rem", "padding": "2rem 1rem"})

@app.callback(
    Output(component_id = "fig-week", component_property = "figure"), 
    Output(component_id = "fig-month", component_property = "figure"), 
    Output(component_id = "fig-importance", component_property = "figure"),
    Input(component_id = "simple-dropdown", component_property = "value")
)
def graph_date_range_show(P): 
    P=4
    model_predict = pd.read_csv("src/model_predict.csv")
    model_importance = pd.read_csv("src/model_feature_importances.csv")
    fig_weekly, fig_monthly = my_plot(date = model_predict.actual_date, true = model_predict.real, pred = model_predict.pred, P = P, shift = False, more = True, main_title = model_name+" ")
    fig_importance = model_feature_importance(model_importance, model_name=model_name, threshold=0, display=True, sort=True, ascending=True, n_feature=30)
    return fig_weekly, fig_monthly, fig_importance

if __name__ == "__main__": 
    app.run_server(debug = False)
    #app.run_server(debug = False, port = 9527, host = "192.168.10.148")