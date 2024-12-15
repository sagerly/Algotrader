from binance.client import Client
import numpy as np
import pandas as pd
import time
import os
from dotenv import load_dotenv
import logging
from datetime import datetime
import random

# Constants
SIDE_BUY = 'BUY'
SIDE_SELL = 'SELL'
ORDER_TYPE_LIMIT = 'LIMIT'
TIME_IN_FORCE_GTC = 'GTC'

class GridBot:
    def __init__(self, symbol, grid_levels=10, grid_size_percent=1.0, test_mode=True, initial_capital=1000):
        # First set the symbol
        self.symbol = symbol

        self.max_drawdown = 0.1  # 10% maximum drawdown

        self.initial_portfolio_value = None
        
        # Then setup logging
        self.setup_logging()
        
        # Test mode configuration
        self.test_mode = test_mode
        if test_mode:
            self.logger.info("Running in TEST MODE - No real trades will be made")
            self.test_balance_base = 0.0  # Start with no BTC
            self.test_balance_quote = initial_capital  # Use provided initial capital
            # Now get_binance_price can access self.symbol
            self.test_price = self.get_binance_price()  # Get actual current price
            self.test_price_movement = 0
            self.client = Client("", "", tld='us')
            self.logger.info(f"Test Mode Initial Capital: ${initial_capital}")
        else:
            # Load real API credentials
            load_dotenv()
            api_key = os.getenv('BINANCE_US_API_KEY')
            api_secret = os.getenv('BINANCE_US_API_SECRET')
            if not api_key or not api_secret:
                raise ValueError("API key and secret required for live trading. Add them to .env file.")
            self.client = Client(api_key, api_secret, tld='us')
        
        self.grid_levels = grid_levels
        self.grid_size_percent = grid_size_percent
        self.buy_orders = []
        self.sell_orders = []
        self.trades = []
        self.trading_fee = 0.001  # 0.1% trading fee
        self.min_trade_amount = 10  # Minimum trade amount in USDT

        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('trading_bot.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def check_drawdown(self):
        if self.initial_portfolio_value is None:
            base_balance, quote_balance = self.get_account_balance()
            current_price = self.get_current_price()
            self.initial_portfolio_value = quote_balance + (base_balance * current_price)
            return True
        
        current_value = quote_balance + (base_balance * current_price)
        drawdown = (self.initial_portfolio_value - current_value) / self.initial_portfolio_value
        
        return drawdown <= self.max_drawdown

    def get_binance_price(self):
        """Get the actual current price from Binance for initialization"""
        try:
            ticker = Client("", "", tld='us').get_symbol_ticker(symbol=self.symbol)
            return float(ticker['price'])
        except Exception as e:
            self.logger.error(f"Error getting initial price: {e}")
            return 50000  # Fallback price if API call fails
    
    def simulate_price_movement(self):
        """Simulate random price movements for testing"""
        volatility = self.test_price * 0.001  # 0.1% volatility
        # Generate new price movement each time instead of accumulating
        price_change = self.test_price * random.uniform(-0.001, 0.001)
        self.test_price = self.test_price + price_change
        
        # Add mean reversion to prevent long-term drift
        mean_reversion = 0.1 * (self.get_binance_price() - self.test_price)
        self.test_price += mean_reversion
        
        return max(self.test_price, 1)

    def get_current_price(self):
        """Get current price - real or simulated"""
        if self.test_mode:
            price = self.simulate_price_movement()
            if random.random() < 0.1:  # Log occasionally to reduce spam
                self.logger.info(f"Test current price: ${price:.2f}")
            return price
        else:
            try:
                ticker = self.client.get_symbol_ticker(symbol=self.symbol)
                return float(ticker['price'])
            except Exception as e:
                self.logger.error(f"Error getting price: {e}")
                return None

    def get_account_balance(self):
        """Get account balance - real or simulated"""
        if self.test_mode:
            return self.test_balance_base, self.test_balance_quote
        else:
            try:
                account = self.client.get_account()
                base_asset = self.symbol[:-4]
                quote_asset = self.symbol[-4:]
                
                base_balance = float([asset['free'] for asset in account['balances'] 
                                    if asset['asset'] == base_asset][0])
                quote_balance = float([asset['free'] for asset in account['balances'] 
                                     if asset['asset'] == quote_asset][0])
                
                self.logger.info(f"Current balance - {base_asset}: {base_balance:.8f}, {quote_asset}: {quote_balance:.2f}")
                return base_balance, quote_balance
            except Exception as e:
                self.logger.error(f"Error getting balance: {e}")
                return None, None

    def setup_grid(self, current_price):
        """Create grid levels around current price"""
        grid_range = current_price * (self.grid_size_percent / 100)
        price_step = grid_range / self.grid_levels
        
        self.logger.info(f"Grid range: ${grid_range:.2f}")
        self.logger.info(f"Price step: ${price_step:.2f}")
        
        grid_prices = []
        for i in range(-self.grid_levels, self.grid_levels + 1):
            grid_prices.append(current_price + (i * price_step))
            
        sorted_prices = sorted(grid_prices)
        
        self.buy_orders = sorted_prices[:-1]
        self.sell_orders = sorted_prices[1:]
        
        self.logger.info(f"Buy orders: {[f'${p:.2f}' for p in self.buy_orders]}")
        self.logger.info(f"Sell orders: {[f'${p:.2f}' for p in self.sell_orders]}")
        
        return sorted_prices
    
    def calculate_quantity(self, price, is_buy):
        try:
            base_balance, quote_balance = self.get_account_balance()
            if base_balance is None or quote_balance is None:
                return None

            # Check minimum balance requirement first
            if quote_balance < self.min_trade_amount * 2:  # Keep some buffer
                self.logger.warning("Remaining balance too low for trading")
                return None

            if is_buy:
                max_possible = quote_balance / price
                # Use larger portions of available balance (30% instead of 10%)
                quantity = min(max_possible * 0.3, max_possible)
                
                # Ensure minimum trade size
                min_quantity = self.min_trade_amount / price
                quantity = max(min_quantity, quantity)
                
                # Make sure we don't exceed available quote balance
                max_quantity = quote_balance / price
                if quantity > max_quantity:
                    quantity = max_quantity
            else:
                # Increase sell quantity to match buy quantity
                quantity = base_balance * 0.3  # Increased from 0.1
                
                # Ensure minimum trade size
                min_quantity = self.min_trade_amount / price
                quantity = max(min_quantity, quantity)
                
                # Make sure we don't exceed available base balance
                if quantity > base_balance:
                    quantity = base_balance

            # Round to 8 decimal places for BTC
            quantity = round(quantity, 8)

            # Final verification that trade value meets minimum
            if quantity * price < self.min_trade_amount:
                self.logger.warning(f"Order value ${quantity * price:.2f} below minimum trade amount ${self.min_trade_amount}")
                return None

            return quantity

        except Exception as e:
            self.logger.error(f"Error calculating quantity: {e}")
            return None

    def place_order(self, side, price, quantity):
        """Place order with fractional support"""
        if self.test_mode:
            order_value = quantity * price
            
            if side == SIDE_BUY:
                if order_value <= self.test_balance_quote:
                    self.test_balance_quote -= order_value
                    self.test_balance_base += quantity * (1 - self.trading_fee)
                    success = True
                    self.logger.info(f"TEST MODE - Bought {quantity:.8f} BTC for ${order_value:.2f}")
                else:
                    success = False
                    self.logger.warning(f"TEST MODE - Insufficient USDT balance for buy")
            else:  # SELL
                if quantity <= self.test_balance_base:
                    self.test_balance_base -= quantity
                    self.test_balance_quote += order_value * (1 - self.trading_fee)
                    success = True
                    self.logger.info(f"TEST MODE - Sold {quantity:.8f} BTC for ${order_value:.2f}")
                else:
                    success = False
                    self.logger.warning(f"TEST MODE - Insufficient BTC balance for sell")

            if success:
                self.trades.append({
                    'timestamp': datetime.now(),
                    'type': side.lower(),
                    'price': float(price),
                    'quantity': float(quantity),
                    'value': float(quantity) * float(price),
                    'fee': float(quantity) * float(price) * self.trading_fee
                })
                
                self.logger.info(
                    f"TEST MODE - New balances - BTC: {self.test_balance_base:.8f}, "
                    f"USDT: ${self.test_balance_quote:.2f}"
                )
                return {'orderId': f'test_{len(self.trades)}'}
            return None
        else:
            try:
                order = self.client.create_order(
                    symbol=self.symbol,
                    side=side,
                    type=ORDER_TYPE_LIMIT,
                    timeInForce=TIME_IN_FORCE_GTC,
                    quantity=quantity,
                    price=str(price)
                )
                
                self.trades.append({
                    'timestamp': datetime.now(),
                    'type': side.lower(),
                    'price': float(price),
                    'quantity': float(quantity),
                    'value': float(quantity) * float(price),
                    'fee': float(quantity) * float(price) * self.trading_fee
                })
                
                return order
                
            except Exception as e:
                self.logger.error(f"Error placing order: {e}")
                return None

    def monitor_and_replace_orders(self):
        """Monitor price and place orders with improved trading logic"""
        last_price_log = time.time()
        last_balance_log = time.time()
        last_trade_time = None
        self.last_trade_time = None
        min_profit_threshold = 0.005  # 0.5% minimum profit threshold
        min_sell_profit_threshold = min_profit_threshold * 0.8  # 80% of buy threshold
        price_history = []
        min_time_between_trades = 300  # 5 minutes minimum between trades
        
        while True:
            try:
                current_price = self.get_current_price()
                if not current_price:
                    time.sleep(1)
                    continue
                
                # Store price history for volatility calculation
                price_history.append(current_price)
                if len(price_history) > 20:  # Keep last 20 prices
                    price_history.pop(0)
                
                # Calculate volatility
                if len(price_history) >= 20:
                    volatility = np.std(price_history) / np.mean(price_history)
                    min_profit_threshold = max(0.005, volatility)  # Adjust threshold based on volatility
                    min_sell_profit_threshold = min_profit_threshold * 0.8  # Lower threshold for sells
                
                # Periodic logging
                if time.time() - last_price_log > 300:
                    self.logger.info(f"Current price: ${current_price:.2f}")
                    self.logger.info(f"Current volatility: {volatility:.4f}")
                    self.logger.info(f"Buy threshold: {min_profit_threshold:.4f}")
                    self.logger.info(f"Sell threshold: {min_sell_profit_threshold:.4f}")
                    last_price_log = time.time()
                
                if time.time() - last_balance_log > 600:
                    base_balance, quote_balance = self.get_account_balance()
                    portfolio_value = quote_balance + (base_balance * current_price)
                    self.logger.info(f"Portfolio Value: ${portfolio_value:.2f}")
                    last_balance_log = time.time()
                
                if (self.last_trade_time is not None and 
                    time.time() - self.last_trade_time < min_time_between_trades):
                    return False
                
                # Check if enough time has passed since last trade
                if last_trade_time and time.time() - last_trade_time < min_time_between_trades:
                    time.sleep(1)
                    continue
                
                # Check buy orders
                for buy_price in self.buy_orders[:]:
                    if current_price <= buy_price:
                        # Calculate potential profit
                        potential_sell_price = buy_price * (1 + self.grid_size_percent/100)
                        potential_profit = (potential_sell_price - buy_price) / buy_price
                        
                        # Check if we should take the trade
                        if not self.should_take_trade(SIDE_BUY, buy_price):
                            continue
                        
                        # Only buy if profit potential exceeds threshold
                        if potential_profit > min_profit_threshold:
                            quantity = self.calculate_quantity(buy_price, True)
                            if quantity:
                                if self.place_order(SIDE_BUY, buy_price, quantity):
                                    self.buy_orders.remove(buy_price)
                                    new_sell_price = potential_sell_price
                                    self.sell_orders.append(new_sell_price)
                                    last_trade_time = time.time()
                                    self.logger.info(f"Added new sell order at ${new_sell_price:.2f} with potential profit: {potential_profit:.2%}")
                
                # Check sell orders
                for sell_price in self.sell_orders[:]:
                    if current_price >= sell_price:
                        realized_profit = (current_price - sell_price) / sell_price
                        
                        # Check if we should take the trade
                        if not self.should_take_trade(SIDE_SELL, sell_price):
                            continue
                        
                        # More aggressive selling - lower threshold
                        if realized_profit > min_sell_profit_threshold:
                            quantity = self.calculate_quantity(sell_price, False)
                            if quantity:
                                if self.place_order(SIDE_SELL, sell_price, quantity):
                                    self.sell_orders.remove(sell_price)
                                    new_buy_price = sell_price * (1 - self.grid_size_percent/100)
                                    self.buy_orders.append(new_buy_price)
                                    last_trade_time = time.time()
                                    self.logger.info(f"Added new buy order at ${new_buy_price:.2f} with realized profit: {realized_profit:.2%}")
                
                time.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Error in monitor_and_replace_orders: {e}")
                time.sleep(1)

    def should_take_trade(self, side, price):
        """Determine if we should take a trade based on portfolio balance"""
        base_balance, quote_balance = self.get_account_balance()
        total_value = quote_balance + (base_balance * price)
        
        if side == SIDE_BUY:
            # Don't buy if we already have too much BTC
            btc_value = base_balance * price
            if btc_value / total_value > 0.6:  # Max 60% in BTC
                return False
        else:  # SELL
            # Don't sell if we have too little BTC
            if base_balance * price < self.min_trade_amount:
                return False
                
        return True

    def calculate_current_metrics(self):
        """Calculate current performance metrics with improved tracking"""
        try:
            if not self.trades:
                return None
            
            base_balance, quote_balance = self.get_account_balance()
            current_price = self.get_current_price()
            
            if None in (base_balance, quote_balance, current_price):
                return None
            
            # Calculate initial portfolio value
            initial_portfolio_value = self.test_balance_quote  # Your initial capital
            
            # Current portfolio value
            current_portfolio_value = quote_balance + (base_balance * current_price)
            
            # Calculate actual profitable trades by comparing buy/sell pairs
            profitable_trades = 0
            total_profit_loss = 0
            buy_trades = [t for t in self.trades if t['type'] == 'buy']
            sell_trades = [t for t in self.trades if t['type'] == 'sell']
            
            for buy, sell in zip(buy_trades, sell_trades):
                if sell['price'] > buy['price']:
                    profitable_trades += 1
                trade_pl = (sell['price'] - buy['price']) * sell['quantity']
                total_profit_loss += trade_pl
            
            metrics = {
                'total_trades': len(self.trades),
                'buy_trades': len(buy_trades),
                'sell_trades': len(sell_trades),
                'profitable_trades': profitable_trades,
                'total_volume': sum(t['value'] for t in self.trades),
                'total_fees': sum(t['fee'] for t in self.trades),
                'total_profit_loss': total_profit_loss,
                'current_portfolio_value': current_portfolio_value,
                'portfolio_return_percent': ((current_portfolio_value - initial_portfolio_value) / initial_portfolio_value) * 100,
                'base_balance': base_balance,
                'quote_balance': quote_balance,
                'current_price': current_price
            }
            
            self.logger.info("\n=== Current Metrics ===")
            for key, value in metrics.items():
                if isinstance(value, float):
                    self.logger.info(f"{key}: {value:.8f}")
                else:
                    self.logger.info(f"{key}: {value}")
                
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {e}")
            return None

    def run(self):
        """Main method to run the grid trading bot"""
        self.logger.info(f"Starting Grid Trading Bot - {'TEST MODE' if self.test_mode else 'LIVE MODE'}")
        try:
            current_price = self.get_current_price()
            if not current_price:
                self.logger.error("Failed to get current price")
                return
                
            self.logger.info(f"Current price: ${current_price:.2f}")
            
            grid_prices = self.setup_grid(current_price)
            self.logger.info(f"Created grid with {len(grid_prices)} price levels")
            
            # Log initial balances
            base_balance, quote_balance = self.get_account_balance()
            self.logger.info(f"Initial portfolio value: ${quote_balance + (base_balance * current_price):.2f}")
            
            self.monitor_and_replace_orders()
            
            # Call to calculate current metrics at the end of the run
            self.calculate_current_metrics()  # Ensure this is reachable
            
        except Exception as e:
            self.logger.error(f"Error running bot: {e}")
            raise

if __name__ == "__main__":
    # Example usage
    bot = GridBot(
        symbol='BTCUSDT',
        grid_levels=15,
        grid_size_percent=1.0,
        test_mode=True,  # Set to False for live trading
        initial_capital=200  # Set your desired initial capital
    )
    
    try:
        bot.run()
    except KeyboardInterrupt:
        print("\nBot stopped by user")
        final_metrics = bot.calculate_current_metrics()
        if final_metrics:
            print("\n=== Final Bot Metrics ===")
            for key, value in final_metrics.items():
                if isinstance(value, float):
                    print(f"{key}: {value:.8f}")
                else:
                    print(f"{key}: {value}")
    except Exception as e:
        print(f"Bot stopped due to error: {str(e)}")