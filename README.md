# **DAM Forecaster**

**Day Ahead Market (DAM) Forecaster** is a machine learning pipeline developed to forecast wholesale electricity prices for the Greek national grid. It aggregates historical grid load and real-time pricing data to train a robust LightGBM predictive model.


-----
-----


## Key Features
* **Automated Data Integration:** Fetches historical load and DAM prices via the **ENTSO-E API**, and TTF Natural Gas prices via **Yahoo Finance (yfinance)**.
* **Feature Engineering:** Generates time-series features such as 24h/7-day lags, rolling averages, and Greek calendar encoding (weekends/holidays).
* **Strict Validation:** Uses `TimeSeriesSplit` (10 folds) to prevent data leakage and evaluate the model purely on chronological, unseen data.


-----


## Model Performance
* **Dynamic Evaluation:** The pipeline automatically calculates and displays the Mean Absolute Error (MAE) via 10-fold cross-validation based on the user-defined date range (e.g 30.52 €/MWh on the 2022-2024 dataset).
* **Note:** This metric is heavily skewed by the extreme geopolitical volatility of the 2022 energy crisis.


-----


## Data Sources
* **ENTSO-E Transparency Platform:** Historical load and Day-Ahead prices for the Greek bidding zone (EIC: 10YGR-HTSO-----Y). Requires a free API token from [transparency.entsoe.eu](https://transparency.entsoe.eu).
* **Yahoo Finance (`yfinance`):** TTF Natural Gas futures, since Greek electricity prices are frequently set by gas-fired plants, making this a key leading indicator for price shocks.

-----


## Tech Stack & Libraries
| Library | Why it was chosen |
|---|---|
| `lightgbm` | High-performance gradient boosting framework, ideal for fast and accurate tabular data regression. |
| `scikit-learn` | Provides `TimeSeriesSplit` for robust chronological cross-validation without data leakage. |
| `pandas` | Essential for time-series data manipulation, gap interpolation, and lag feature creation. |
| `entsoe-py` | The official API wrapper to reliably fetch Greek grid load and Day-Ahead prices. |
| `yfinance` | Retrieves TTF Natural Gas prices, a critical feature for capturing energy market shocks. |
| `holidays` | Automatically identifies Greek public holidays to help the model understand demand drops. |
| `matplotlib` | Generates clear visual diagnostics comparing actual vs. predicted prices. |
| `python-dotenv` | Industry standard for securely managing API keys locally without hardcoding them. |


-----


## Installation Guide
### Requirements
- Python 3.8+
- Active ENTSO-E API token
### Setup
In your IDE's terminal please enter the following:
```bash
git clone https://github.com/YOUR_USERNAME/dam-forecaster.git
cd dam-forecaster
pip install pandas scikit-learn lightgbm entsoe-py holidays yfinance matplotlib python-dotenv
```
### API Key Configuration
Create a `.env` file in the root directory of the project and add your personal ENTSO-E API key:
```bash
ENTSOE_TOKEN=your_personal_entsoe_api_key_here
```


-----


## License
MIT License: Free to use, modify, and distribute with attribution.


-----
-----


## **Developer**
**Dimitrios Poulos** <br>
*Electrical \& Computer Engineering Student*
* **Release Date:** October 2026
