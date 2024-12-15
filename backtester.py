from binance.client import Client
from binance.enums import *
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

class GridBacktester:
    def __init__(self, symbol, start_date, end_date, initial_capital, grid_levels=10, grid_size_percent=1.0):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.grid_levels = grid_levels
        self.grid_size_percent = grid_size_percent
        self.trades = []
        self.portfolio_value = []
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('trading_bot.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize client with Binance.US
        self.client = Client("", "", tld='us')
        
    def get_historical_data(self):
        """Fetch historical price data from Binance"""
        try:
            self.logger.info(f"Fetching historical data for {self.symbol} from {self.start_date} to {self.end_date}")
            klines = self.client.get_historical_klines(
                self.symbol,
                Client.KLINE_INTERVAL_5MINUTE,
                self.start_date,
                self.end_date
            )
            
            if not klines:
                self.logger.error("No data received from Binance")
                return None
                
            self.logger.info(f"Received {len(klines)} data points")
            
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 
                'volume', 'close_time', 'quote_volume', 'trades_count',
                'taker_buy_volume', 'taker_buy_quote_volume', 'ignore'
            ])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp')
            
            # Convert price columns to float
            for col in ['open', 'high', 'low', 'close']:
                df[col] = df[col].astype(float)
                
            self.logger.info("Data processing completed successfully")
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching historical data: {str(e)}")
            self.logger.exception("Full traceback:")
            return None
            
    def setup_grid(self, current_price):
        """Create grid levels around the current price"""
        grid_range = current_price * (self.grid_size_percent / 100)
        price_step = grid_range / self.grid_levels

        self.logger.info(f"Grid range: ${grid_range:.2f}")
        self.logger.info(f"Price step: ${price_step:.2f}")
        
        grid_prices = []
        for i in range(-self.grid_levels, self.grid_levels + 1):
            grid_prices.append(current_price + (i * price_step))
            
        return sorted(grid_prices)
        
    def simulate_trades(self, df):
        """Simulate grid trading strategy"""
        capital = self.initial_capital
        base_currency = 0  # Amount of cryptocurrency held
        quote_currency = capital  # Amount of USDT held
        
        # Initialize grid
        current_price = df['close'].iloc[0]
        grid_prices = self.setup_grid(current_price)

        self.logger.info(f"Initial price: ${current_price:.2f}")
        self.logger.info(f"Grid prices: {[f'${p:.2f}' for p in grid_prices]}")
        self.logger.info(f"Initial capital: ${quote_currency:.2f}")

        buy_orders = [price for price in grid_prices[:-1]]  # Lower prices for buy orders
        sell_orders = [price for price in grid_prices[1:]]  # Higher prices for sell orders
        
        for timestamp, row in df.iterrows():
            low, high = row['low'], row['high']
            
            # Check for filled buy orders
            for buy_price in buy_orders[:]:
                if low <= buy_price <= high and quote_currency > 0:
                    quantity = (quote_currency * 0.1) / buy_price  # Use 10% of available quote currency
                    quote_currency -= quantity * buy_price
                    base_currency += quantity * 0.999  # Account for 0.1% trading fee
                    
                    self.trades.append({
                        'timestamp': timestamp,
                        'type': 'buy',
                        'price': buy_price,
                        'quantity': quantity,
                        'value': quantity * buy_price
                    })

                    self.logger.info(f"Buy executed: {quantity:.8f} BTC at ${buy_price:.2f}")

                    
                    # Remove filled order and add new sell order above
                    buy_orders.remove(buy_price)
                    new_sell_price = buy_price * (1 + self.grid_size_percent/100)
                    sell_orders.append(new_sell_price)
            
            # Check for filled sell orders
            for sell_price in sell_orders[:]:
                if low <= sell_price <= high and base_currency > 0:
                    quantity = base_currency * 0.1  # Sell 10% of available base currency
                    quote_currency += quantity * sell_price * 0.999  # Account for 0.1% trading fee
                    base_currency -= quantity
                    
                    self.trades.append({
                        'timestamp': timestamp,
                        'type': 'sell',
                        'price': sell_price,
                        'quantity': quantity,
                        'value': quantity * sell_price
                    })

                    self.logger.info(f"Sell executed: {quantity:.8f} BTC at ${sell_price:.2f}")

                    # Remove filled order and add new buy order below
                    sell_orders.remove(sell_price)
                    new_buy_price = sell_price * (1 - self.grid_size_percent/100)
                    buy_orders.append(new_buy_price)
            
            # Record portfolio value
            portfolio_value = quote_currency + (base_currency * row['close'])
            self.portfolio_value.append({
                'timestamp': timestamp,
                'value': portfolio_value
            })
            
    def calculate_metrics(self):
        """Calculate performance metrics"""
        if not self.trades or not self.portfolio_value:
            return None
            
        # Convert trades and portfolio value to DataFrames
        trades_df = pd.DataFrame(self.trades)
        portfolio_df = pd.DataFrame(self.portfolio_value)
        
        # Calculate basic metrics
        total_trades = len(trades_df)
        profitable_trades = len(trades_df[trades_df['type'] == 'sell'])
        
        initial_value = self.initial_capital
        final_value = portfolio_df['value'].iloc[-1]
        total_return = ((final_value - initial_value) / initial_value) * 100
        
        # Calculate daily returns
        portfolio_df['daily_return'] = portfolio_df['value'].pct_change()
        
        # Calculate Sharpe Ratio (assuming risk-free rate of 2%)
        risk_free_rate = 0.02
        sharpe_ratio = np.sqrt(365) * (portfolio_df['daily_return'].mean() - risk_free_rate/365) / portfolio_df['daily_return'].std()
        
        # Calculate Maximum Drawdown
        portfolio_df['cummax'] = portfolio_df['value'].cummax()
        portfolio_df['drawdown'] = (portfolio_df['cummax'] - portfolio_df['value']) / portfolio_df['cummax']
        max_drawdown = portfolio_df['drawdown'].max()
        
        # Calculate win rate
        win_rate = len(trades_df[trades_df['type'] == 'sell']) / (total_trades/2)
        
        return {
            'total_trades': total_trades,
            'profitable_trades': profitable_trades,
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'final_value': final_value
        }
        
    def run_backtest(self):
        """Run the complete backtest"""
        try:
            self.logger.info("Starting backtest")
            
            # Get historical data
            df = self.get_historical_data()
            if df is None:
                self.logger.error("Failed to get historical data")
                return None
                
            # Run simulation
            self.logger.info("Running trade simulation")
            self.simulate_trades(df)
            
            # Calculate and return metrics
            self.logger.info("Calculating metrics")
            metrics = self.calculate_metrics()
            
            if metrics:
                self.logger.info("Backtest completed successfully")
            else:
                self.logger.error("Failed to calculate metrics")
                
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error in run_backtest: {str(e)}")
            return None

if __name__ == "__main__":
    # Create backtester instance
    backtester = GridBacktester(
        symbol='BTCUSDT',
        start_date="2024-06-01",
        end_date="2024-12-01",
        initial_capital=500,
        grid_levels=10,
        grid_size_percent=10.0
    )
    
    # Run backtest and get results
    try:
        results = backtester.run_backtest()
        
        if results:
            print("\n=== Backtest Results ===")
            print(f"Total Return: {results['total_return']:.2f}%")
            print(f"Win Rate: {results['win_rate']*100:.2f}%")
            print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
            print(f"Maximum Drawdown: {results['max_drawdown']*100:.2f}%")
            print(f"Total Trades: {results['total_trades']}")
            print(f"Profitable Trades: {results['profitable_trades']}")
            print(f"Final Portfolio Value: ${results['final_value']:.2f}")
        else:
            print("No results were generated. Check the logs for errors.")
    except Exception as e:
        print(f"Error during backtest execution: {str(e)}")