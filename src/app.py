import streamlit as st
import pandas as pd
import numpy as np
import datetime
import plotly
import plotly.offline as pyoff
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from joblib import dump, load
from prophet import Prophet
from datetime import datetime, timedelta
from mlforecast import MLForecast
from numba import jit
from awesome_streamlit.shared import components
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


#To get the data
df = pd.read_parquet('traffic_10lines.parquet')

#The App title
st.title('Traffic Forecaster')

HOME_AIRPORTS = ('LGW', 'LIS', 'SSA', 'NTE', 'LYS', 'PNH', 'POP', 'SCL')

# Build the selection of the Paired airport based on the Home airport in order to avoid the selection of inexisting routes
with st.sidebar:
    HOME_AIRPORTS = st.selectbox('Home Airport :airplane_departure:', HOME_AIRPORTS)
    if HOME_AIRPORTS == "LGW":
        PAIRED_AIRPORTS = ("AMS", "BCN")
    elif HOME_AIRPORTS  == "LIS":
        PAIRED_AIRPORTS = ("ORY", "OPO")
    elif HOME_AIRPORTS  == "SSA":
        PAIRED_AIRPORTS = ("GRU",)
    elif HOME_AIRPORTS  == "NTE":
        PAIRED_AIRPORTS = ("FUE",)  
    elif HOME_AIRPORTS  == "LYS":
        PAIRED_AIRPORTS = ("PIS",)
    elif HOME_AIRPORTS  == "PNH":
        PAIRED_AIRPORTS = ("NGB" ,)
    elif HOME_AIRPORTS  == "POP":
        PAIRED_AIRPORTS = ("JFK" ,)
    elif HOME_AIRPORTS  == "SCL":
        PAIRED_AIRPORTS = ("LHR" ,)
    
    PAIRED_AIRPORTS = st.selectbox('Paired Airport :airplane_arriving:', PAIRED_AIRPORTS)
    
    forecast_date = st.date_input('Forecast Start Date')
    model_selection = st.selectbox('Model Selection', ['Prophet', 'LGBMRegressor', 'XGBRegressor', 'RandomForestRegressor'])
    nb_days = st.slider('Days of forecast', 7, 30, 1)
    run_forecast = st.button('Forecast')
    
st.write('Home Airport selected:', HOME_AIRPORTS)
st.write('Paired Airport selected:', PAIRED_AIRPORTS)
st.write('Model selected:', model_selection)
st.write('Days of forecast:', nb_days)
st.write('Date selected:', forecast_date)



#import datetime
#import plotly

#import plotly.offline as pyoff
#import plotly.graph_objs as go

#from plotly.subplots import make_subplots


# To compute the performances of each model
def performances(y, y_hat):
    delta_y = np.square(y - y_hat)
    mse = np.nanmean(delta_y)
    rmse = np.sqrt(np.nanmean(delta_y))
    absolute_diff = np.abs(delta_y)
    mae = np.mean(absolute_diff)
    return mse, rmse, mae


# The Dataframe by route selected
air_route_df = df[(df['home_airport'] == HOME_AIRPORTS) & (df['paired_airport'] == PAIRED_AIRPORTS)]
# The Visualization of the Original Dataframe by route selected
st.subheader("Original DataFrame :floppy_disk:")
st.dataframe(data=air_route_df, width=600, height=300)

#The Visualization of the chart by route selected
air_route_df_pre = df.query(f'home_airport == "{HOME_AIRPORTS}" and paired_airport == "{PAIRED_AIRPORTS}"')
air_route_df_pre = air_route_df_pre.groupby('date').agg(pax_total=('pax', 'sum')).reset_index()
fig = make_subplots(rows=1, cols=1)
fig.add_trace(go.Scatter(x=air_route_df_pre['date'], y=air_route_df_pre['pax_total'], fill='tozeroy', name=f'{HOME_AIRPORTS} \U00002708 {PAIRED_AIRPORTS}'), row=1, col=1)
graph_label = f'Route selected : {HOME_AIRPORTS} \U00002708 {PAIRED_AIRPORTS}'
fig.update_layout(title=graph_label)

mae = 0.0
rmse = 0.0
r_squared = 0.0

if run_forecast:
    # Build route traffic dataframe
    air_route_df = df.query(f'home_airport == "{HOME_AIRPORTS}" and paired_airport == "{PAIRED_AIRPORTS}"')
    air_route_df = air_route_df.groupby('date').agg(pax_total=('pax', 'sum')).reset_index()
    # the period considered
    forecast_dates = pd.date_range(forecast_date, periods=nb_days)  
    if model_selection == 'Prophet':
        # Prepare the data for prophet
        forecast_date = pd.to_datetime(forecast_date)
        air_route_df = air_route_df[air_route_df['date'] <= forecast_date]
        prophet_df = air_route_df[['date', 'pax_total']].rename(columns={'date': 'ds', 'pax_total': 'y'})
        # Create the model
        model_prophet = Prophet()
        # Fit the model
        model_prophet.fit(prophet_df)
        # Prediction using the forecast dates
        future = pd.DataFrame({'ds': forecast_dates})
        forecast = model_prophet.predict(future)
        # Create the forecast dataframe
        forecast_df = pd.DataFrame({'date': forecast_dates, 'pax_total': forecast['yhat']})
        # Compute the performance of the prediction
        true_values = air_route_df['pax_total'].values[-nb_days:]
        predicted_values = forecast['yhat'].values
        mse, rmse, mae = performances(true_values, predicted_values)
        # Unpdate the graph with the forecast
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', line=dict(dash='dash', color='yellow'), name='Prophet'), row=1, col=1)

    elif model_selection == 'XGBRegressor':
        # Prepare the data for the XGBRegressor
        ref_date = np.min(air_route_df['date']).to_pydatetime()
        X_train = (air_route_df['date'] - ref_date).dt.days.values.reshape(-1, 1)
        X_train_forecast = (forecast_dates - ref_date).days.to_numpy().reshape(-1, 1)
        y_train = air_route_df['pax_total'].values
        # Data Preprocessing
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_train_forecast_scaled = scaler.transform(X_train_forecast)
        # Hyperparameter Tuning
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 4, 5]
        }     
        # Create the model
        xgb_model = XGBRegressor()        
        grid_search = GridSearchCV(xgb_model, param_grid, cv=5)        
        # Fit the model
        #xgb_model.fit(X_train, y_train)        
        grid_search.fit(X_train_scaled, y_train)
        best_model = grid_search.best_estimator_        
        # Prediction using the forecast dates
        #xgb_predictions = xgb_model.predict(X_train_forecast)        
        xgb_predictions = best_model.predict(X_train_forecast_scaled)        
        # Create the forecast dataframe
        forecast_df = pd.DataFrame({'date': forecast_dates, 'pax_total': xgb_predictions})
        # Compute the performance of the prediction
        true_values = air_route_df['pax_total'].values[-nb_days:]
        predicted_values = forecast_df['pax_total'].values
        mse, rmse, mae = performances(true_values, predicted_values)
        # Unpdate the graph with the forecast
        fig.add_trace(go.Scatter(x=forecast_dates, y=xgb_predictions, mode='lines', line=dict(dash='dash', color='yellow'),
                       name='XGBRegressor'), row=1, col=1)

    elif model_selection == 'RandomForestRegressor':
        # Prepare the data for the RandomForestRegressor
        ref_date = np.min(air_route_df['date']).to_pydatetime()
        X_train = (air_route_df['date'] - ref_date).dt.days.values.reshape(-1, 1)
        X_train_forecast = (forecast_dates - ref_date).days.to_numpy().reshape(-1, 1)
        y_train = air_route_df['pax_total'].values
        # Data Preprocessing
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_train_forecast_scaled = scaler.transform(X_train_forecast)
        # Hyperparameter Tuning
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 5, 10],
            'min_samples_split': [2, 5, 10]
        }
        # Create the model
        rf_model = RandomForestRegressor()
        grid_search = GridSearchCV(rf_model, param_grid, cv=5)
        # Fit the model
        #rf_model.fit(X_train, y_train)
        grid_search.fit(X_train_scaled, y_train)
        best_model = grid_search.best_estimator_
        # Fit the model
        best_model.fit(X_train_scaled, y_train)

        # Prediction using the forecast dates
        #rf_predictions = rf_model.predict(X_train_forecast)
        rf_predictions = best_model.predict(X_train_forecast_scaled)

        # Create the forecast dataframe
        forecast_df = pd.DataFrame({'date': forecast_dates, 'pax_total': rf_predictions})
        # Compute the performance of the prediction
        true_values = air_route_df['pax_total'].values[-nb_days:]
        predicted_values = forecast_df['pax_total'].values
        mse, rmse, mae = performances(true_values, predicted_values)
        # Unpdate the graph with the forecast
        fig.add_trace(go.Scatter(x=forecast_dates, y=rf_predictions, mode='lines', line=dict(dash='dash', color='yellow'),
                       name='RandomForestRegressor'), row=1, col=1)
        
    elif model_selection == 'LGBMRegressor':
        # Prepare the data for the LGBMRegressor
        ref_date = np.min(air_route_df['date']).to_pydatetime()
        X_train = (air_route_df['date'] - ref_date).dt.days.values.reshape(-1, 1)
        X_train_forecast = (forecast_dates - ref_date).days.to_numpy().reshape(-1, 1)
        y_train = air_route_df['pax_total'].values
        # Create the model
        lgb_model = LGBMRegressor()
        # Fit the model
        lgb_model.fit(X_train, y_train)        
        # Prediction using the forecast dates
        lgb_predictions = lgb_model.predict(X_train_forecast)        
        # Create the forecast dataframe
        forecast_df = pd.DataFrame({'date': forecast_dates, 'pax_total': lgb_predictions})
        # Compute the performance of the prediction
        true_values = air_route_df['pax_total'].values[-nb_days:]
        mse, rmse, mae = performances(true_values, lgb_predictions)
        # Unpdate the graph with the forecast
        fig.add_trace(go.Scatter(x=forecast_dates, y=lgb_predictions, mode='lines', line=dict(dash='dash', color='yellow'),
                       name='LGBMRegressor'), row=1, col=1)
      
    # Unpdate the graph with the performances value as label
    graph_label += f"<br>MSE: {mse:.2f} - RMSE: {rmse:.2f} - <br>MAE: {mae:.2f}"
    fig.update_layout(title=graph_label)
    
# Show the praph with the forecast
st.plotly_chart(fig)
# Visualize the Forecast dataframe
if run_forecast:
    st.subheader("Forecast DataFrame :male_mage:")
    st.dataframe(data=forecast_df, width=600, height=300)
    
    
