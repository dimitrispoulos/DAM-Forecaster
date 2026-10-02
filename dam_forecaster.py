"""
    Project: DAM Forecaster
    Description: Trains a machine learning model (LightGBM) to forecast wholesale electricity prices (Day-Ahead Market) for the Greek national grid. Aggregates historical load data, DAM prices, and natural gas (TTF) prices in order to estimate future prices.
    Author: Dimitrios Poulos
"""



import os

import pandas as pd    # Used for data manipulation and analysis.
from entsoe import EntsoePandasClient    # Official API client to retrieve grid data from the ENTSO-E platform.
from sklearn.model_selection import TimeSeriesSplit
from lightgbm import LGBMRegressor    # Importing LightGBM Regressor for model training.
import holidays
import yfinance as yf    # Used to fetch financial data, specifically for retrieving natural gas prices (TTF) which influence electricity prices in Greece.
import matplotlib.pyplot as plt

from dotenv import load_dotenv    # Used to load environment variables from a .env file.
load_dotenv()    # Load environment variables from .env file



# API Configuration
ENTSOE_token = os.environ.get('ENTSOE_TOKEN')    # ENTSO-E API token
ENTSOE_client = EntsoePandasClient(api_key=ENTSOE_token)
EIC_GR = "10YGR-HTSO-----Y"    # EIC code for Greece


# Define the time period (Data prior to 2020 is not comparable due to market structure changes)
start_date = pd.Timestamp('2022-01-01', tz='Europe/Athens')    # Start date for data retrieval
end_date = pd.Timestamp('2025-05-30', tz='Europe/Athens')    # End date for data retrieval

print(f"Fetching data from {start_date} to {end_date} for Greece")



def fetch_load_in_chunks(client, country, start, end):
    """
    Fetches load data in monthly chunks to prevent API timeouts from ENTSO-E.
    
    Arguments:
        client (EntsoePandasClient): The ENTSO-E API connection object.
        country (str): The country code (e.g., 'GR' for Greece).
        start (pd.Timestamp): The initial search date.
        end (pd.Timestamp): The final search date.
        
    Returns:
        pd.DataFrame: A unified DataFrame containing the load data for the entire selected period.
    """
    chunks = []
    current_start = start
    while current_start < end:
        # Calculation of the end of the current month (or the final end_date if less than a month remains)
        current_end = min(current_start + pd.DateOffset(months=1), end)
        print(f"Fetching load data from {current_start.date()} to {current_end.date()}...")
        
        try:
            data = client.query_load(country, start=current_start, end=current_end)
            chunks.append(data)
        except Exception as e:
            print(f"Error occurred while fetching data from {current_start.date()}: {e}")
            
        current_start = current_end
        
    # Union of all months into a single DataFrame
    return pd.concat(chunks)

# Calls the new function instead of the simple query_load
load_data = fetch_load_in_chunks(ENTSOE_client, 'GR', start_date, end_date)
day_ahead_data = ENTSOE_client.query_day_ahead_prices(EIC_GR, start=start_date, end=end_date)    # Fetching day-ahead prices for Greece from ENTSO-E


greek_holidays = holidays.Greece(years=[2021, 2022, 2023, 2024, 2025])    # Fetching Greek holidays for the specified years



# Combine load and price into a single pandas DataFrame
df = pd.DataFrame({
    'Load (MW)': load_data['Actual Load'],    # Load data in megawatts
    'Day-Ahead Price (€/MWh)': day_ahead_data    # Day-ahead price data in euros per megawatt-hour
})
df = df.interpolate(method='linear')    # Interpolating missing values in the DataFrame using linear interpolation

print(df.head())    # Displaying the first few rows of the DataFrame

# Natural gas price (TTF). Greek electricity prices are often set by gas-fired plants, so this feature specifically targets energy-shock periods.
gas_price_data = yf.download('TTF=F', start=start_date.tz_localize(None), end=end_date.tz_localize(None))
gas_price_daily = gas_price_data['Close']
gas_price_daily.index = gas_price_daily.index.tz_localize('Europe/Athens')
df['GasPriceTTF'] = gas_price_daily.reindex(df.index, method='ffill')    # TTF only has daily prices, so we forward-fill it across every hour of the same day



df['Hour'] = df.index.hour
df['DayOfWeek'] = df.index.dayofweek
df['IsWeekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
df['IsHoliday'] = pd.Series(df.index.date, index=df.index).isin(greek_holidays).astype(int)
# Lag Features
df['PriceLag24Hours'] = df['Day-Ahead Price (€/MWh)'].shift(24)    # Lagging the price by 24 hours
df['LoadLag24Hours'] = df['Load (MW)'].shift(24)    # Lagging the load by 24 hours
df['PriceLag7Days'] = df['Day-Ahead Price (€/MWh)'].shift(168)    # Lagging the price by 7 days (168 hours)
df['LoadLag7Days'] = df['Load (MW)'].shift(168)    # Lagging the load by 7 days (168 hours)
df['PriceRolling24h'] = df['Day-Ahead Price (€/MWh)'].shift(1).rolling(window=24).mean()   # Calculating the rolling mean of the price over the past 24 hours
df = df.dropna()    # Dropping rows with NaN values




# Model Preparation
# X: Input features (The variables the model will use to make the prediction)
X = df[['Hour', 'DayOfWeek', 'IsWeekend', 'IsHoliday', 'PriceLag24Hours', 'LoadLag24Hours', 'PriceLag7Days', 'LoadLag7Days', 'PriceRolling24h', 'GasPriceTTF']]
# y: The day-ahead price we want to predict
y = df['Day-Ahead Price (€/MWh)']

# TimeSeriesSplit instead of a plain train_test_split: respects chronological order and gives multiple folds for a more reliable performance estimate than a single, potentially lucky/unlucky split.
time_series_split = TimeSeriesSplit(n_splits=10)    # Initializing a time series split for cross-validation with 10 folds
mae_scores = []    # List to store mean absolute error scores for each fold

all_actual_prices = []    # List to store all actual prices across folds
all_predicted_prices = []    # List to store all predicted prices across folds

print("\nTraining of the model...")
for fold_number, (train_indices, test_indices) in enumerate(time_series_split.split(X)):
    X_train = X.iloc[train_indices]    # X_train: Input features for the training set
    X_test = X.iloc[test_indices]    # X_test: Input features for the testing set
    y_train = y.iloc[train_indices]    # y_train: Actual prices for the training set
    y_test = y.iloc[test_indices]    # y_test: Actual prices for the testing set

    model = LGBMRegressor(
        n_estimators=500,
        objective='mae',
        learning_rate=0.03,
        num_leaves=31,
        max_depth=7,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )
    model.fit(X_train, y_train)
    predicted_prices = model.predict(X_test)

    fold__mae = (y_test - predicted_prices).abs().mean()    # Calculating the mean absolute error for the current fold
    mae_scores.append(fold__mae)    # Adding the mean absolute error to the list
    print(f"Fold {fold_number+1}: MAE = {fold__mae:.2f} €/MWh  (test period: {X_test.index.min().date()} έως {X_test.index.max().date()})")

    all_actual_prices.append(y_test)    # Storing the actual prices for later plotting
    all_predicted_prices.append(pd.Series(predicted_prices, index=y_test.index))    # Storing the predicted prices as a Series with dates
    
average_mae = sum(mae_scores) / len(mae_scores)    # Calculating the average mean absolute error across all folds
print(f"\nAverage MAE across all folds: {average_mae:.2f} €/MWh")

final_actual_prices = pd.concat(all_actual_prices)    # Concatenating all actual prices into a single series
final_predicted_prices = pd.concat(all_predicted_prices)    # Concatenating all predicted prices into a single series


last24h_prices = final_predicted_prices.tail(24)    # Getting the last 24 hours of predicted prices
dam_export = pd.DataFrame({
    "hour": range(24),
    "predicted_price": last24h_prices.values
})
dam_export.to_csv("dam_forecast.csv", index=False)    # Exporting the last 24 hours of predicted prices to a CSV file
print("\nLast 24 hours of predicted prices exported to 'dam_forecast.csv'")    # Informing the user that the export was successful




print("\nPlotting the results...")
plt.figure(figsize=(15, 7))

plt.plot(final_actual_prices.index, final_actual_prices, label='Actual Day-Ahead Price', color='blue')    # Plotting the actual day-ahead prices
plt.plot(final_predicted_prices.index, final_predicted_prices, label='Predicted Day-Ahead Price', color='red', linestyle='--')    # Plotting the predicted day-ahead prices
plt.title('Day-Ahead Price Prediction for Greece')    # Setting the title of the plot
plt.grid(visible=True)
plt.xlabel('Date')    # Setting the x-axis label
plt.ylabel('Price (€/MWh)')    # Setting the y-axis label
plt.legend()    # Displaying the legend
plt.savefig('dam_forecast.png')    # Saving the plot to a file
plt.show()    # Showing the plot