
import pandas as pd
import plotly.graph_objects as go

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

def trend_check(x):
    if x != x:
       return x
    elif x >= 0:
       return "+"
    else:
       return "-"

def my_plot_weekly(date, true, pred, P, shift, more, main_title): 
    
    trend_true = (true - true.shift(1)).map(lambda x: trend_check(x))
    trend_pred = (pred - pd.Series(pred).shift(1)).map(lambda x: trend_check(x))
    trend_index = trend_true.notnull() & trend_pred.notnull()

    trend_true = trend_true[trend_index]
    trend_pred = trend_pred[trend_index]
    trend_match = trend_true == trend_pred
    trend_accuracy = sum(trend_match) / len(trend_true)
    
    ll = len(date)
    pp = true.isnull().sum()
    
    mae = mean_absolute_error(true[true.notnull()], pred[true.notnull()])
    rmse = mean_squared_error(true[true.notnull()], pred[true.notnull()], squared = False)
    
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x = date, 
            y = true, 
            mode = "lines", 
            name = "NEWC", 
            line = dict(color = "#0066CC")
        )
    )
    
    if more: 
        fig.add_trace(
            go.Scatter(
                x = date, 
                y = pred, 
                mode = "lines", 
                name = "P-" + str(P) + " Forecast", 
                line = dict(color = "#FF8040", dash = "dot")
            )
        )
        # trend point
        fig.add_trace(
            go.Scatter(
                x = date[trend_match[trend_match].index], 
                y = pred[trend_match[trend_match].index], 
                mode = "markers", 
                name = "Correct", 
                marker = dict(symbol = 28, color = "green")
            )
        )
        fig.add_trace(
            go.Scatter(
                x = date[trend_match[-trend_match].index], 
                y = pred[trend_match[-trend_match].index], 
                mode = "markers", 
                name = "Wrong", 
                marker = dict(symbol = 4, color = "red")
            )
        )
    else: 
        fig.add_trace(
            go.Scatter(
                x = date, 
                y = pred, 
                mode = "lines", 
                name = "P-" + str(P) + " Forecast", 
                line = dict(color = "#FF8040", dash = "dot")
            )
        )
    
    if shift: 
        fig.add_trace(
            go.Scatter(
                x = date[:ll-pp], 
                y = pred[pp:], 
                mode = "lines", 
                name = "Shift Back", 
                line = dict(color = "black", dash = "dot")
            )
        )
    fig.update_layout(
        title = main_title + str(P) + 
        "-Week Model, MAE: " + str(round(mae, 2)) + 
        "; RMSE: " + str(round(rmse, 2)) + 
        "; Trend Accuarcy: " + str(round(trend_accuracy, 2)), 
        margin = dict(l = 0, r = 0, t = 30, b = 0)
    )
    return fig

def my_plot_monthly(date, true, pred, P, more, main_title): 
    
    df_month = pd.DataFrame({
        "actual_date": date, 
        "true": true, 
        "pred": pred
    })
    df_month["actual_ym"] = [d[:7] for d in df_month.actual_date]
    df_month = df_month.groupby(["actual_ym"]).mean().reset_index(drop = False)
    
    trend_true = (df_month.true - df_month.true.shift(1)).map(lambda x: trend_check(x))
    trend_pred = (df_month.pred - df_month.pred.shift(1)).map(lambda x: trend_check(x))
    trend_index = trend_true.notnull() & trend_pred.notnull()

    trend_true = trend_true[trend_index]
    trend_pred = trend_pred[trend_index]
    trend_match = trend_true == trend_pred
    trend_accuracy = sum(trend_match) / len(trend_true)
    
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x = df_month.actual_ym, 
            y = df_month.true, 
            mode = "lines", 
            name = "NEWC", 
            line = dict(color = "#0066CC")
        )
    )
    
    if more: 
        fig.add_trace(
            go.Scatter(
                x = df_month.actual_ym, 
                y = df_month.pred, 
                mode = "lines", 
                name = "P-" + str(P) + " Forecast", 
                line = dict(color = "#FF8040", dash = "dot")
            )
        )
        # trend point
        fig.add_trace(
            go.Scatter(
                x = df_month.actual_ym[trend_match[trend_match].index], 
                y = df_month.pred[trend_match[trend_match].index], 
                mode = "markers", 
                name = "Correct", 
                marker = dict(symbol = 28, color = "green", size = 10)
            )
        )
        fig.add_trace(
            go.Scatter(
                x = df_month.actual_ym[trend_match[-trend_match].index], 
                y = df_month.pred[trend_match[-trend_match].index], 
                mode = "markers", 
                name = "Wrong", 
                marker = dict(symbol = 4, color = "red", size = 10)
            )
        )
    else: 
        fig.add_trace(
            go.Scatter(
                x = df_month.actual_ym, 
                y = df_month.pred, 
                mode = "lines", 
                name = "P-" + str(P) + " Forecast", 
                line = dict(color = "#FF8040", dash = "dot")
            )
        )
    
    fig.update_layout(
        title = main_title + "Monthly, Trend Acc: " + str(round(trend_accuracy, 2)), 
        margin = dict(l = 0, r = 0, t = 30, b = 0)
    )
    return fig

def my_plot(date, true, pred, P, shift = False, more = True, main_title = ""): 
    fig_weekly = my_plot_weekly(date, true, pred, P, shift, more, main_title)
    fig_monthly = my_plot_monthly(date, true, pred, P, more, main_title)
    return fig_weekly, fig_monthly

def model_feature_importance(df, model_name= "RandomForestRegressor",threshold=0.3, display=True, sort=True, ascending=True, n_feature=30):
    if sort:
        df = df.sort_values(by='value', ascending=ascending)
    if n_feature > 0:
        df = df.tail(n_feature)
    if threshold != None:
        df = df[df['value'] >= threshold]
    if display:       
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=df['feature'],
            x=df['value'],
            name='feature importance',
            orientation='h',
            marker=dict(
                color='rgba(47, 142, 238, 0.6)'
            )
        ))
        fig.update_layout(
            title = "{} Feature Importance".format(model_name), 
            title_x=0.5)

    return fig