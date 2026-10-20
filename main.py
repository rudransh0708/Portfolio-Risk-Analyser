import os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# --------------------------------
# 1. Portfolio Configuration
# --------------------------------

tickers = ["AAPL", "MSFT", "NVDA", "JPM", "KO"]

weights = np.array([0.30, 0.25, 0.20, 0.15, 0.10])

assert np.isclose(weights.sum(), 1.0), "Portfolio weights must sum to 1"

# --------------------------------
# 2. Historical Data Collection
# --------------------------------

# Create data folder if it doesn't exist
os.makedirs("data", exist_ok=True)

file_path = "data/historical_prices.csv"

# Check whether historical data already exists
if os.path.exists(file_path):

    print("Loading historical prices from saved CSV...")

    prices = pd.read_csv(
        file_path,
        index_col=0,
        parse_dates=True
    )

else:

    print("Downloading historical stock prices...")

    data = yf.download(
        tickers,
        period="5y",
        auto_adjust=True,
        progress=False
    )

    # Extract adjusted closing prices
    prices = data["Close"]

    # Match portfolio weight order
    prices = prices[tickers]

    # Remove missing values
    prices = prices.dropna()

    # Save dataset
    prices.to_csv(file_path)

    print("Historical prices saved successfully!")

# --------------------------------
# 3. Data Validation
# --------------------------------

# Ensure the correct column order
prices = prices[tickers]

# Remove any rows with missing values
prices = prices.dropna()

# Check dataset
assert not prices.empty, "No valid price data available"

assert not prices.isna().any().any(), "Dataset contains missing values"

# --------------------------------
# 4. Display Historical Data
# --------------------------------

print("\nFirst five rows:")
print(prices.head())

print("\nLast five rows:")
print(prices.tail())

print("\nDataset shape:")
print(prices.shape)

print("\nMissing values:")
print(prices.isna().sum())

print("\nPortfolio weights:")
for ticker, weight in zip(tickers, weights):
    print(f"{ticker}: {weight:.0%}")

# --------------------------------
# 5. Individual Stock Daily Returns
# --------------------------------

daily_returns = prices.pct_change().dropna()

print("\nIndividual Stock Daily Returns:")
print(daily_returns.head())


# --------------------------------
# 6. Portfolio Daily Returns
# --------------------------------

portfolio_returns = daily_returns.dot(weights)

print("\nPortfolio Daily Returns:")
print(portfolio_returns.head())


# --------------------------------
# 7. Cumulative Portfolio Returns
# --------------------------------

cumulative_returns = (1 + portfolio_returns).cumprod() - 1

print("\nFinal Cumulative Portfolio Return:")
print(f"{cumulative_returns.iloc[-1]:.2%}")


# --------------------------------
# 8. Annualized Portfolio Return
# --------------------------------

trading_days = 252

total_days = len(portfolio_returns)

annualized_return = (
    (1 + cumulative_returns.iloc[-1]) ** (trading_days / total_days)
) - 1

print("\nAnnualized Portfolio Return:")
print(f"{annualized_return:.2%}")


# --------------------------------
# 9. Plot Cumulative Portfolio Returns
# --------------------------------

plt.figure(figsize=(12, 6))

plt.plot(
    cumulative_returns.index,
    cumulative_returns * 100,
    label="Portfolio"
)

plt.title("Cumulative Portfolio Returns")
plt.xlabel("Date")
plt.ylabel("Cumulative Return (%)")

plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

os.makedirs("figures", exist_ok=True)

plt.savefig("figures/portfolio_cumulative_returns.png")

plt.show()