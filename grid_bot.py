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
    def __init__(self, symbol, grid_levels=15, grid_size_percent=0.5, test_mode=True, initial_capital=200):
        # First set core parameters
        self.symbol = symbol
        self.initial_capital = initial_capital
        self.grid_levels = grid_levels
        self.grid_size_percent = grid_size_percent
        self.test_mode = test_mode
        self.initial_grid_state = None
        
        # Risk management parameters
        self.max_drawdown = 0.05  # 10% maximum drawdown
        self.initial_portfolio_value = None
        self.trading_fee = 0.001  # 0.1% trading fee
        self.min_trade_amount = 10  # Minimum trade amount in USDT
        
        # Initialize trading containers
        self.buy_orders = []
        self.sell_orders = []
        self.trades = []

        self.price_history = []
        
        # Setup logging
        self.setup_logging()
        
        # Test mode configuration
        self.test_mode = test_mode
        if test_mode:
            self.logger.info("Running in TEST MODE - No real trades will be made")
            self.test_balance_base = 0.0  # Start with no BTC
            self.test_balance_quote = initial_capital  # Use provided initial capital
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
    
    def update_grid_levels(self, current_price):
        """Update grid when price moves beyond optimal range"""
        grid_min = min(self.buy_orders)
        grid_max = max(self.sell_orders)
        
        # For BTC, use tighter thresholds (15% instead of 25%)
        if (current_price < grid_min * 0.85 or 
            current_price > grid_max * 1.15):
            
            self.logger.info(f"Price ${current_price:.2f} outside grid range (${grid_min:.2f} - ${grid_max:.2f})")
            self.logger.info("Recentering grid...")
            
            # Store performance metrics before resetting
            if len(self.trades) > 0:
                self.calculate_current_metrics()
                
            # Setup new grid
            self.setup_grid(current_price)
            return True
        return False
    
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
        """Create grid levels around current price with improved distribution"""
        try:
            # Validate input
            if not current_price or current_price <= 0:
                self.logger.error("Invalid current price for grid setup")
                return None

            # Calculate grid parameters
            grid_range = current_price * (self.grid_size_percent / 100)
            step_size = grid_range / (self.grid_levels * 2.5)
            
            # Log grid parameters
            self.logger.info(f"Setting up grid around ${current_price:.2f}")
            self.logger.info(f"Grid range: ${grid_range:.2f} (±{self.grid_size_percent/2}%)")
            self.logger.info(f"Step size: ${step_size:.2f}")

            # Create buy orders below current price
            self.buy_orders = []
            for i in range(1, self.grid_levels + 1):
                buy_price = current_price - (i * step_size)
                # Ensure minimum price is reasonable
                if buy_price > current_price * 0.5:  # Don't go below 50% of current price
                    self.buy_orders.append(round(buy_price, 2))

            # Create sell orders above current price
            self.sell_orders = []
            for i in range(1, self.grid_levels + 1):
                sell_price = current_price + (i * step_size)
                # Ensure maximum price is reasonable
                if sell_price < current_price * 1.5:  # Don't go above 150% of current price
                    self.sell_orders.append(round(sell_price, 2))

            # Sort orders
            self.buy_orders = sorted(self.buy_orders, reverse=True)  # Highest to lowest
            self.sell_orders = sorted(self.sell_orders)  # Lowest to highest

            # Calculate expected profits for each grid level
            for i, buy_price in enumerate(self.buy_orders):
                if i < len(self.sell_orders):
                    sell_price = self.sell_orders[i]
                    profit_potential = ((sell_price - buy_price) / buy_price) * 100
                    self.logger.info(
                        f"Grid level {i+1}: Buy ${buy_price:.2f} -> Sell ${sell_price:.2f} "
                        f"(Potential profit: {profit_potential:.2f}%)"
                    )

            # Validate grid setup
            if not self.buy_orders or not self.sell_orders:
                self.logger.error("Failed to create valid grid levels")
                return None

            # Log grid summary
            self.logger.info(f"Created grid with {len(self.buy_orders)} buy levels and {len(self.sell_orders)} sell levels")
            self.logger.info(f"Buy range: ${min(self.buy_orders):.2f} to ${max(self.buy_orders):.2f}")
            self.logger.info(f"Sell range: ${min(self.sell_orders):.2f} to ${max(self.sell_orders):.2f}")

            # Return all price levels for reference
            all_prices = sorted(self.buy_orders + self.sell_orders + [current_price])
            
            # Store initial grid state for reference
            self.initial_grid_state = {
                'timestamp': datetime.now(),
                'current_price': current_price,
                'buy_orders': self.buy_orders.copy(),
                'sell_orders': self.sell_orders.copy(),
                'grid_range': grid_range,
                'step_size': step_size
            }

            return all_prices

        except Exception as e:
            self.logger.error(f"Error setting up grid: {e}")
            return None
    
    def analyze_market_conditions(self, price_history):
        """Analyze if market conditions are suitable for trading"""
        if not price_history or len(price_history) < 5:  # Check if price_history is empty
            return False
        volatility = np.std(price_history) / np.mean(price_history)

        # Only trade if market is trending (not choppy)
        return volatility > 0.0003

    def calculate_quantity(self, price, is_buy):
        """Calculate order quantity with improved position sizing and detailed logging"""
        try:
            base_balance, quote_balance = self.get_account_balance()
            if base_balance is None or quote_balance is None:
                self.logger.error("Failed to get account balances")
                return None

            # Log initial state
            self.logger.info(f"Calculate quantity for {'BUY' if is_buy else 'SELL'} @ ${price:.2f}")
            self.logger.info(f"Current balances - BTC: {base_balance:.8f}, USDT: ${quote_balance:.2f}")

            if is_buy:
                # Calculate maximum available funds (increased from 10% to 25% of available balance)
                max_funds_percent = 0.25
                available_funds = min(quote_balance, self.initial_capital * max_funds_percent)
                
                # Make sure we have enough for minimum trade
                if available_funds < self.min_trade_amount:
                    self.logger.warning(
                        f"Available funds ${available_funds:.2f} below minimum ${self.min_trade_amount}"
                    )
                    return None

                # Calculate quantity based on available funds
                quantity = available_funds / price
                
                # Round to 8 decimal places for BTC
                quantity = round(quantity, 8)
                
                # Calculate final order value
                order_value = quantity * price
                
                # Log calculation details
                self.logger.info(
                    f"Buy calculation:"
                    f"\n- Max funds (25%): ${self.initial_capital * max_funds_percent:.2f}"
                    f"\n- Available USDT: ${available_funds:.2f}"
                    f"\n- Calculated quantity: {quantity:.8f} BTC"
                    f"\n- Order value: ${order_value:.2f}"
                )

                # Verify final order value meets minimum
                if order_value < self.min_trade_amount:
                    self.logger.warning(
                        f"Final order value ${order_value:.2f} below minimum ${self.min_trade_amount}"
                    )
                    return None

                return quantity

            else:  # SELL
                # Use 25% of available BTC for each sell
                max_quantity = base_balance * 0.25
                
                # Make sure selling this amount would exceed minimum trade value
                order_value = max_quantity * price
                if order_value < self.min_trade_amount:
                    self.logger.warning(
                        f"Sell value ${order_value:.2f} below minimum ${self.min_trade_amount}"
                    )
                    return None

                # Round to 8 decimal places
                quantity = round(max_quantity, 8)
                
                # Log calculation details
                self.logger.info(
                    f"Sell calculation:"
                    f"\n- Available BTC: {base_balance:.8f}"
                    f"\n- 25% of BTC: {max_quantity:.8f}"
                    f"\n- Final quantity: {quantity:.8f}"
                    f"\n- Order value: ${quantity * price:.2f}"
                )

                return quantity

        except Exception as e:
            self.logger.error(f"Error in calculate_quantity: {e}")
            return None

    def place_order(self, side, price, quantity):
        """Place order with fractional support and improved error handling"""
        try:
            # Validate inputs
            if quantity is None or price is None or quantity <= 0 or price <= 0:
                self.logger.error("Invalid quantity or price")
                return None

            order_value = quantity * price
            
            if self.test_mode:
                if side == SIDE_BUY:
                    # Check if we have enough USDT for the buy
                    if order_value <= self.test_balance_quote:
                        # Calculate fee
                        fee = order_value * self.trading_fee
                        total_cost = order_value + fee
                        
                        # Execute buy
                        if total_cost <= self.test_balance_quote:
                            self.test_balance_quote -= total_cost
                            self.test_balance_base += quantity
                            success = True
                            self.logger.info(
                                f"TEST MODE - Bought {quantity:.8f} BTC for ${order_value:.2f} "
                                f"(fee: ${fee:.2f})"
                            )
                        else:
                            success = False
                            self.logger.warning("TEST MODE - Insufficient USDT balance including fees")
                    else:
                        success = False
                        self.logger.warning("TEST MODE - Insufficient USDT balance for buy")
                
                else:  # SELL
                    # Check if we have enough BTC for the sell
                    if quantity <= self.test_balance_base:
                        # Calculate fee
                        fee = order_value * self.trading_fee
                        net_proceeds = order_value - fee
                        
                        # Execute sell
                        self.test_balance_base -= quantity
                        self.test_balance_quote += net_proceeds
                        success = True
                        self.logger.info(
                            f"TEST MODE - Sold {quantity:.8f} BTC for ${order_value:.2f} "
                            f"(fee: ${fee:.2f})"
                        )
                    else:
                        success = False
                        self.logger.warning("TEST MODE - Insufficient BTC balance for sell")

                if success:
                    # Record the trade
                    trade = {
                        'timestamp': datetime.now(),
                        'type': side.lower(),
                        'price': float(price),
                        'quantity': float(quantity),
                        'value': float(order_value),
                        'fee': float(order_value) * self.trading_fee,
                        'net_value': float(order_value) * (1 - self.trading_fee)
                    }
                    self.trades.append(trade)
                    
                    # Log updated balances
                    self.logger.info(
                        f"TEST MODE - New balances - BTC: {self.test_balance_base:.8f}, "
                        f"USDT: ${self.test_balance_quote:.2f}"
                    )
                    
                    # Calculate and log profit/loss if it's a sell
                    if side == SIDE_SELL and len(self.trades) > 1:
                        last_buy = next(
                            (t for t in reversed(self.trades[:-1]) if t['type'] == 'buy'), 
                            None
                        )
                        if last_buy:
                            pl = trade['net_value'] - last_buy['value']
                            pl_percent = (pl / last_buy['value']) * 100
                            self.logger.info(
                                f"TEST MODE - Trade P/L: ${pl:.2f} ({pl_percent:.2f}%)"
                            )
                
                    return {'orderId': f'test_{len(self.trades)}'}
                return None

            else:
                # Live trading mode
                try:
                    # Convert price to string with appropriate precision
                    price_str = f"{price:.2f}"
                    
                    # Place the order
                    order = self.client.create_order(
                        symbol=self.symbol,
                        side=side,
                        type=ORDER_TYPE_LIMIT,
                        timeInForce=TIME_IN_FORCE_GTC,
                        quantity=quantity,
                        price=price_str
                    )
                    
                    # Record the trade
                    self.trades.append({
                        'timestamp': datetime.now(),
                        'type': side.lower(),
                        'price': float(price),
                        'quantity': float(quantity),
                        'value': float(quantity) * float(price),
                        'fee': float(quantity) * float(price) * self.trading_fee,
                        'order_id': order['orderId']
                    })
                    
                    self.logger.info(
                        f"Order placed - {side} {quantity:.8f} BTC at ${price:.2f}"
                    )
                    
                    return order
                    
                except Exception as e:
                    self.logger.error(f"Error placing live order: {e}")
                    return None

        except Exception as e:
            self.logger.error(f"Unexpected error in place_order: {e}")
            return None

    def should_take_trade(self, side, price):
        """More selective trade entry conditions with improved risk management"""
        current_time = time.time()
        
        # Reduce cooldown period from 300 to 60 seconds
        if hasattr(self, 'last_trade_time') and \
           current_time - self.last_trade_time < 60:  # Changed from 300
            return False
        
        # Check market conditions
        base_balance, quote_balance = self.get_account_balance()
        total_value = quote_balance + (base_balance * price)
        
        # Relax stop loss from 5% to 7%
        if total_value < self.initial_capital * 0.93:  # Changed from 0.95
            self.logger.warning(f"Stop loss triggered. Total value: ${total_value:.2f}")
            return False
        
        self.logger.info(f"Current portfolio - BTC: {base_balance:.8f}, USDT: ${quote_balance:.2f}")
        
        # Reduce required price history
        if len(self.price_history) < 3:  # Changed from 5
            self.logger.info("Not enough price history for analysis")
            return False
        
        if side == SIDE_BUY:
            # Relax BTC position limit from 40% to 50%
            btc_value = base_balance * price
            if btc_value / total_value > 0.5:  # Changed from 0.4
                self.logger.info(f"Skip buy: BTC position ({btc_value/total_value:.2%}) exceeds max 50%")
                return False
            
            # Reduce minimum USDT requirement
            if quote_balance < self.min_trade_amount * 1.5:  # Changed from 2
                self.logger.info(f"Skip buy: Insufficient USDT balance (${quote_balance:.2f})")
                return False
            
            # Relax local bottom requirement
            price_min = min(self.price_history[-5:])  # Look at fewer prices
            if price > price_min * 1.01:  # Changed from 1.005
                self.logger.info("Skip buy: Price not near local bottom")
                return False
            
        else:  # SELL
            if base_balance * price < self.min_trade_amount:
                self.logger.info(f"Skip sell: BTC position too small (${base_balance * price:.2f})")
                return False
            
            # Relax local top requirement
            price_max = max(self.price_history[-5:])  # Look at fewer prices
            if price < price_max * 0.99:  # Changed from 0.997
                self.logger.info("Skip sell: Price not near local top")
                return False
        
        # Reduce volatility requirement
        if len(self.price_history) >= 12:  # Changed from 24
            daily_volatility = np.std(self.price_history[-12:]) / np.mean(self.price_history[-12:])
            if daily_volatility < 0.0005:  # Changed from 0.001
                self.logger.info(f"Skip trade: Low volatility ({daily_volatility:.4f})")
                return False
        
        return True

    def monitor_and_replace_orders(self):
        """Monitor price and place orders with more selective trading conditions"""
        last_price_log = time.time()
        last_balance_log = time.time()
        last_trade_time = None
        min_profit_threshold = 0.0005  # Changed from 0.001
        min_volatility_threshold = 0.0002  # Changed from 0.0003
        consolidation_period = 6  # Changed from 12
        min_time_between_trades = 60  # Changed from 180
        
        while True:
            try:
                current_price = self.get_current_price()
                if not current_price:
                    time.sleep(1)
                    continue

                # Store price history
                self.price_history.append(current_price)
                if len(self.price_history) > consolidation_period:
                    self.price_history.pop(0)
                
                # Only consider trading if we have enough price history
                if len(self.price_history) >= consolidation_period:
                    volatility = np.std(self.price_history[-consolidation_period:]) / np.mean(self.price_history[-consolidation_period:])
                    price_trend = (self.price_history[-1] - self.price_history[0]) / self.price_history[0]
                    
                    # Relax trend requirements
                    for buy_price in self.buy_orders[:]:
                        if current_price <= buy_price:
                            if price_trend < -0.0003:  # Changed from -0.0005
                                potential_profit = (buy_price * (1 + self.grid_size_percent/100) - buy_price) / buy_price
                                if potential_profit > min_profit_threshold and self.should_take_trade(SIDE_BUY, buy_price):
                                    quantity = self.calculate_quantity(buy_price, True)
                                    if quantity:
                                        if self.place_order(SIDE_BUY, buy_price, quantity):
                                            self.buy_orders.remove(buy_price)
                                            new_sell_price = buy_price * (1 + self.grid_size_percent/100)
                                            self.sell_orders.append(new_sell_price)
                                            last_trade_time = time.time()
                                            self.last_trade_time = time.time()
                
                    for sell_price in self.sell_orders[:]:
                        if current_price >= sell_price:
                            if price_trend > 0.0003:  # Changed from 0.0005
                                realized_profit = (current_price - sell_price) / sell_price
                                if realized_profit > min_profit_threshold and self.should_take_trade(SIDE_SELL, sell_price):
                                    quantity = self.calculate_quantity(sell_price, False)
                                    if quantity:
                                        if self.place_order(SIDE_SELL, sell_price, quantity):
                                            self.sell_orders.remove(sell_price)
                                            new_buy_price = sell_price * (1 - self.grid_size_percent/100)
                                            self.buy_orders.append(new_buy_price)
                                            last_trade_time = time.time()
                                            self.last_trade_time = time.time()
                
                time.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Error in monitor_and_replace_orders: {e}")
                time.sleep(1)

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
            
            # Ensure monitoring starts correctly
            self.monitor_and_replace_orders()  # Ensure this is called directly
            
            # Call to calculate current metrics after monitoring
            self.calculate_current_metrics()  # Added this line to ensure the method is reachable
            
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