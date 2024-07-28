import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from datetime import date, timedelta

import BrownianMotion

GRAPH_DAYS = 500
MAX_DAYS_STOCK_HISTORY = 10

class Stock:
    def __init__(self, ticker, date=str(date.today())):
        self.ticker = ticker
        self.stock = yf.Ticker(self.ticker)
        self.as_of_date = date
        self.price = self.get_close_price(date)

        # Try to add sector if it exists
        try:
            self.sector = self.stock.info['sector']
        except KeyError:
            self.sector = None

    def __str__(self):
        closing_prices = self.get_close_prices(np.datetime64(self.as_of_date), GRAPH_DAYS)
        
        plt.figure(figsize=(9, 6))
        plt.plot(closing_prices.index, closing_prices, label="Price")
        plt.xlabel("Date")
        plt.ylabel("Closing Price")
        plt.title(f"{self.stock.info.get('longName')} Stock Price (Past {GRAPH_DAYS} Days)")
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
        return "\n"*3

    def payoff(self, price):
        return price - self.strike
    
    def get_return(self, start_date, end_date, all_prices=None, all_prices_path=None):
        try:
            start_price = self.get_close_price(start_date, all_prices, all_prices_path)
            end_price = self.get_close_price(end_date, all_prices, all_prices_path)
            return end_price / start_price - 1
        except KeyError:
            print(f"Check {self.ticker} between {start_date} and {end_date}")
            return 0

    def get_close_prices(self, end_date, no_days):
        stock_df = self.stock.history(start=str(np.datetime64(end_date) - no_days), end=str(end_date), interval='1d')
        return stock_df[['Close']].rename(columns={'Close': 'Price'})
    
    def get_daily_returns(self, end_date, no_days):
        close_prices = self.get_close_prices(end_date, no_days)
        return close_prices.pct_change().dropna()

    def get_close_price(self, date, all_prices=None, all_prices_path=None):
        if all_prices:
            # Check if price is already downloaded
            if self.ticker in all_prices.columns and date in all_prices['Date'].values:
                price = all_prices[all_prices['Date'] == date][self.ticker].values[0]
                if not pd.isna(price):
                    return price
            
            # Otherwise download it
            date = np.datetime64(date)
            stock_df = self.stock.history(start=str(date-MAX_DAYS_STOCK_HISTORY), end=str(date), interval = "1d").reset_index()
            price = stock_df['Close'].loc[0]
            self.save_stock_price(date, price, all_prices_path)
            return price
        
        else:
            stock_df = self.get_close_prices(date, MAX_DAYS_STOCK_HISTORY)
            # Return price if it exists
            try:
                return stock_df.loc[stock_df.index.max(), 'Price']
            except KeyError:
                return 0
      
    def save_stock_price(self, date_str, price, all_prices_path):
        all_prices = pd.read_csv(all_prices_path)
        if self.ticker not in all_prices.columns:
            all_prices.insert(loc=len(all_prices.columns), column=self.ticker, value=np.nan)
        if date_str not in all_prices['Date'].values:
            all_prices.loc[len(all_prices.index)] = np.nan
            all_prices.loc[len(all_prices.index)-1, 'Date'] = date_str
        
        all_prices.loc[all_prices['Date'] == date_str, [self.ticker]] = price
        all_prices.sort_values(by=['Date'])
        all_prices.to_csv(all_prices_path, index=False)

    def mu(self, no_days):
        daily_returns = self.get_daily_returns(self.as_of_date, no_days)
        return daily_returns.mean().iloc[0]
    
    def vol(self, no_days):
        daily_returns = self.get_daily_returns(self.as_of_date, no_days)
        no_trading_days = len(daily_returns.index)
        return np.sqrt(daily_returns.var() * np.sqrt(no_trading_days)).iloc[0]

def plot_stock_and_gbm(stock, gbm):
    # Get historical prices
    closing_prices = stock.get_close_prices(np.datetime64(stock.as_of_date), GRAPH_DAYS)
    
    # Get GBM paths
    gbm_paths = gbm.get_paths()
    last_date = closing_prices.index[-1]
    future_times = [last_date + timedelta(days=int(t * 365)) for t in gbm.times]
    
    # Plot historical prices
    plt.figure(figsize=(12, 8))
    plt.plot(closing_prices.index, closing_prices['Price'], label='Historical Prices', color='blue')

    # Plot GBM paths
    for i in range(gbm.num_paths):
        plt.plot(future_times, gbm_paths[:, i], color='cornflowerblue', lw=0.5)

    # Define custom legend entries
    historical_line = Line2D([0], [0], color='blue', lw=2, label='Historical Prices')
    gbm_line = Line2D([0], [0], color='cornflowerblue', lw=0.5, label='Simulated GBM Paths')
    
    # Add title, labels, and legend
    plt.title(f"{stock.stock.info.get('longName')} Stock Price (Historical and Simulated)")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend(handles=[historical_line, gbm_line], loc='best')
    
    plt.tight_layout()
    plt.show()

def main():
    stock = Stock('AAPL')
    brownian_motion = BrownianMotion.GeometricBrownianMotion(stock.price, stock.mu(365), stock.vol(365), 1)
    plot_stock_and_gbm(stock, brownian_motion)

if __name__ == '__main__':
    main()