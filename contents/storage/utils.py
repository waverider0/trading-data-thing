import os
import shutil
import requests
import datetime as dt

import pandas as pd
import yfinance as yf

# Set dirname to folder containing this file
dirname = os.path.dirname(__file__)


def delete_temp_models(model_name):
    if os.listdir(f'{dirname}/temp'):
        if os.path.exists(f'{dirname}/temp/{model_name}'):
            shutil.rmtree(f'{dirname}/temp/{model_name}')

def get_saved_models():
    """ Returns a list of all the saved models """
    saved_models = []
    if os.listdir(f'{dirname}/saved_models'):
        for file in os.listdir(f'{dirname}/saved_models'):
            saved_models.append(file)
    return saved_models

def get_macro_data():
    start = '2003-01-01'
    end = dt.datetime.now()

    fred_apikey = '4d68b01a5415707192c45e61f180299d'
    fred_api_endpoint = 'https://api.stlouisfed.org/fred/series/observations'

    fred_series_codes = [
        'GDPC1', # Real Gross Domestic Product
        'DFF', # Effective Federal Funds Rate (avg of daily rates)
        'FEDFUNDS', # Effective Federal Funds Rate (set by the fed)
        'BOPGSTB', # Balance of Payments
        'DGS1', # 1-Year Treasury Bond Yield
        'DGS2', # 2-Year Treasury Bond Yield
        'DGS3', # 3-Year Treasury Bond Yield
        'DGS5', # 5-Year Treasury Bond Yield
        'DGS7', # 7-Year Treasury Bond Yield
        'DGS10', # 10-Year Treasury Bond Yield
        'DGS20', # 20-Year Treasury Bond Yield
        'DGS30', # 30-Year Treasury Bond Yield
        'UNRATE', # Unemployment Rate
        'PAYEMS', # All Employees: Total Nonfarm Payrolls
        'PPIACO', # All Commodities: All Items PPI
        'CPALTT01USM659N', # All Commodities: All Items CPI
        'CORESTICKM159SFRBATL', # Core Personal Consumption Expenditures Price Index
        'UMCSENT', # Consumer Sentiment
        'PCECC96', # Personal Consumption Expenditures: Chain-type Price Index
        'PCEPI', # Personal Consumption Expenditures: Price Index
        'HOUST', # Total New Private Housing Units Started
        'HNFSUSNSA', # New One Family Houses for Sale
        'RHORUSQ156N', # Homeownership Rate
        'TOTALSA', # Total Vehicle Sales
        'RETAILMPCSMSA', # Total retail sales
        'RSXFS', # Retail Sales: Food Services and Drinking Places
        'WILL5000PR', # Wilshire 5000 Price Index
    ]
    yfinance_tickers = [
        'ES=F', # S&P 500
        'NQ=F', # Nasdaq 100
        'YM=F', # Dow Jones Industrial Average
        'ZB=F', # 30-Year Treasury Bond
        'ZN=F', # 10-Year Treasury Bond
        'ZF=F', # 5-Year Treasury Bond
        'ZT=F', # 2-Year Treasury Bond
        'GC=F', # Gold
        'SI=F', # Silver
        'HG=F', # Copper
        'CL=F', # Crude Oil
        'HO=F', # Heating Oil
        'NG=F', # Natural Gas
        'RB=F', # RBOB Gasoline
        'ZC=F', # Corn
        'ZO=F', # Oats
        'KE=F', # KC HRW Wheat
        'ZR=F', # Rough Rice
        'ZS=F', # Soybeans
        'ZM=F', # Soybean Meal
        'HE=F', # Lean Hogs
        'LE=F', # Live Cattle
        'CC=F', # Cocoa
        'KC=F', # Coffee
        'CT=F', # Cotton
        'LBS=F', # Lumber
        'SB=F', # Sugar
        '^VIX', # CBOE Volatility Index
    ]

    # Get data from yfinance and fred
    yfinance_data = pd.DataFrame()
    fred_data = pd.DataFrame()
    for ticker in yfinance_tickers:
        yfinance_data[ticker] = yf.download(ticker, start=start, end=end)['Close']
    for series_id in fred_series_codes:
        params = {
            'series_id': series_id,
            'api_key': fred_apikey,
            'file_type': 'json',
            'limit': 100000
        }
        response = requests.get(fred_api_endpoint, params=params)
        data = pd.DataFrame(response.json()['observations']).drop(columns=['realtime_start', 'realtime_end']).set_index('date')
        data = data.set_index(pd.to_datetime(data.index))
        data = data.rename(columns={'value': series_id})
        fred_data = pd.concat([fred_data, data], axis=1)

    # Interpolate missing values
    futures_data = yfinance_data.interpolate(method='linear', limit_direction ='forward')
    fred_data = fred_data.ffill()

    # Concatenate dataframes
    df = pd.concat([futures_data, fred_data], axis=1)
    df = df.loc[start:end]
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.dropna()

    # Save to csv
    df.to_csv(f'{dirname}/macro_data.csv')