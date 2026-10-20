
import os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# Portfolio setup
tickers = ["AAPL", "MSFT", "NVDA", "JPM", "KO"]
weights = np.array([0.30, 0.25, 0.20, 0.15, 0.10])

assert np.isclose(weights.sum(), 1.0), "Portfolio weights must sum to 1"

# Load saved prices or download them
os.makedirs("data", exist_ok=True)
file_path = "data/historical_prices.csv"

if os.path.exists(file_path):
    print("Loading historical prices from saved CSV...")
    prices = pd.read_csv(file_path, index_col=0, parse_dates=True)

else:
    print("Downloading historical stock prices...")

    data = yf.download(
        tickers,
        period="5y",
        auto_adjust=True,
        progress=False
    )

    prices = data["Close"]
    prices = prices[tickers].dropna()
    prices.to_csv(file_path)

    print("Historical prices saved successfully!")

# Clean and check the data
prices = prices[tickers].dropna()

assert not prices.empty, "No valid price data available"
assert not prices.isna().any().any(), "Dataset contains missing values"
assert len(prices) > 1, "Not enough observations to calculate returns"

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

# Calculate daily stock and portfolio returns
daily_returns = prices.pct_change().dropna()
portfolio_returns = daily_returns.dot(weights)

print("\nIndividual Stock Daily Returns:")
print(daily_returns.head())

print("\nPortfolio Daily Returns:")
print(portfolio_returns.head())

# Calculate cumulative and annualized returns
cumulative_returns = (1 + portfolio_returns).cumprod() - 1

trading_days = 252
total_days = len(portfolio_returns)

annualized_return = (
    (1 + cumulative_returns.iloc[-1]) ** (trading_days / total_days)
) - 1

print("\nFinal Cumulative Portfolio Return:")
print(f"{cumulative_returns.iloc[-1]:.2%}")

print("\nAnnualized Portfolio Return:")
print(f"{annualized_return:.2%}")

# Save the cumulative returns graph
os.makedirs("figures", exist_ok=True)

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
plt.savefig("figures/portfolio_cumulative_returns.png")
plt.show()

# Calculate individual stock volatility
daily_volatility = daily_returns.std()
annual_volatility = daily_volatility * np.sqrt(trading_days)

print("\nAnnualized Stock Volatility:")
for ticker, volatility in annual_volatility.items():
    print(f"{ticker}: {volatility:.2%}")

# Calculate the covariance matrix
daily_cov_matrix = daily_returns.cov()
annual_cov_matrix = daily_cov_matrix * trading_days

print("\nAnnualized Covariance Matrix:")
print(annual_cov_matrix.round(4))

# Calculate correlations between stocks
correlation_matrix = daily_returns.corr()

print("\nStock Correlation Matrix:")
print(correlation_matrix.round(3))

# Plot the correlation heatmap
fig, ax = plt.subplots(figsize=(9, 7))

heatmap = ax.imshow(
    correlation_matrix,
    cmap="coolwarm",
    vmin=-1,
    vmax=1
)

ax.set_xticks(range(len(tickers)))
ax.set_yticks(range(len(tickers)))
ax.set_xticklabels(tickers)
ax.set_yticklabels(tickers)

for i in range(len(tickers)):
    for j in range(len(tickers)):
        ax.text(
            j, i,
            f"{correlation_matrix.iloc[i, j]:.2f}",
            ha="center",
            va="center",
            color="black"
        )

fig.colorbar(heatmap, ax=ax, label="Correlation")
ax.set_title("Stock Return Correlation Matrix")

plt.tight_layout()
plt.savefig("figures/portfolio_correlation_matrix.png")
plt.show()

# Calculate portfolio volatility using the covariance matrix
portfolio_variance = (
    weights.T @ annual_cov_matrix.to_numpy() @ weights
)

portfolio_volatility = np.sqrt(portfolio_variance)

print("\nPortfolio Annualized Variance:")
print(f"{portfolio_variance:.6f}")

print("\nPortfolio Annualized Volatility:")
print(f"{portfolio_volatility:.2%}")

# Check the result using daily portfolio returns
direct_portfolio_volatility = (
    portfolio_returns.std() * np.sqrt(trading_days)
)

assert np.isclose(
    portfolio_volatility,
    direct_portfolio_volatility
), "Portfolio volatility calculations do not match"

print("\nVolatility from Daily Portfolio Returns:")
print(f"{direct_portfolio_volatility:.2%}")

# Measure the reduction in volatility from diversification
weighted_average_volatility = np.dot(weights, annual_volatility)

diversification_benefit = (
    weighted_average_volatility - portfolio_volatility
)

print("\nWeighted Average Stock Volatility:")
print(f"{weighted_average_volatility:.2%}")

print("\nActual Portfolio Volatility:")
print(f"{portfolio_volatility:.2%}")

print("\nDiversification Benefit:")
print(f"{diversification_benefit:.2%}")
