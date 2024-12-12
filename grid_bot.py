from binance.client import Client
from binance.enums import *
import numpy as np
import time
import os
from dotenv import load_dotenv
import logging
from datetime import datetime

class EnhancedGridBot:
    def __init__(self, symbol, grid_levels=10, grid_size_percent=1.0, copy_trader_id=None):
        # Setup logging
        self.setup_logging()
        
        # Load API credentials
        load_dotenv()
        api_key = os.getenv('BINANCE_API_KEY')
        api_secret = os.getenv('BINANCE_API_SECRET')
        
        # Initialize Binance client
        self.client = Client(api_key, api_secret)
        self.symbol = symbol
        self.grid_levels = grid_levels
        self.grid_size_percent = grid_size_percent
        self.active_orders = {}
        self.copy_trader_id = copy_trader_id
        self.last_copy_check = datetime.now()
        
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
    
    def get_current_price(self):
        """Get current price of trading pair"""
        try:
            ticker = self.client.get_symbol_ticker(symbol=self.symbol)
            return float(ticker['price'])
        except Exception as e:
            self.logger.error(f"Error getting price: {e}")
            return None
            
    def setup_grid(self):
        """Create grid levels around current price"""
        current_price = self.get_current_price()
        if not current_price:
            return None
            
        grid_range = current_price * (self.grid_size_percent / 100)
        price_step = grid_range / self.grid_levels
        
        grid_prices = []
        for i in range(-self.grid_levels, self.grid_levels + 1):
            grid_prices.append(current_price + (i * price_step))
            
        return sorted(grid_prices)
    
    def copy_trader_orders(self):
        """Copy orders from successful trader if enabled"""
        if not self.copy_trader_id:
            return
            
        try:
            # Check if enough time has passed since last copy (e.g., 5 minutes)
            if (datetime.now() - self.last_copy_check).total_seconds() < 300:
                return
                
            # Get trader's recent trades
            trades = self.client.get_recent_trades(symbol=self.symbol, limit=10)
            
            for trade in trades:
                if str(trade['buyer']) == self.copy_trader_id:
                    # Copy buy order
                    self.place_copy_order(trade, SIDE_BUY)
                elif str(trade['seller']) == self.copy_trader_id:
                    # Copy sell order
                    self.place_copy_order(trade, SIDE_SELL)
                    
            self.last_copy_check = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Error copying trades: {e}")
    
    def place_copy_order(self, trade, side):
        """Place an order copying another trader's position"""
        try:
            quantity = float(trade['qty'])
            price = float(trade['price'])
            
            order = self.client.create_order(
                symbol=self.symbol,
                side=side,
                type=ORDER_TYPE_LIMIT,
                timeInForce=TIME_IN_FORCE_GTC,
                quantity=quantity,
                price=str(price)
            )
            
            self.logger.info(f"Copied {side} order: {quantity} {self.symbol} at {price}")
            return order
            
        except Exception as e:
            self.logger.error(f"Error placing copy order: {e}")
            return None
    
    def calculate_quantity(self, price):
        """Calculate order quantity based on minimum notional value"""
        try:
            min_notional = 10  # Minimum order value in USDT
            quantity = min_notional / price
            
            # Get symbol info for precision
            info = self.client.get_symbol_info(self.symbol)
            step_size = float([f for f in info['filters'] if f['filterType'] == 'LOT_SIZE'][0]['stepSize'])
            precision = int(round(-np.log10(step_size)))
            
            return round(quantity, precision)
            
        except Exception as e:
            self.logger.error(f"Error calculating quantity: {e}")
            return None
    
    def place_grid_orders(self, grid_prices):
        """Place buy and sell orders at grid levels"""
        for i in range(len(grid_prices) - 1):
            try:
                quantity = self.calculate_quantity(grid_prices[i])
                if not quantity:
                    continue
                    
                # Place buy order
                buy_order = self.client.create_order(
                    symbol=self.symbol,
                    side=SIDE_BUY,
                    type=ORDER_TYPE_LIMIT,
                    timeInForce=TIME_IN_FORCE_GTC,
                    quantity=quantity,
                    price=str(grid_prices[i])
                )
                self.active_orders[buy_order['orderId']] = buy_order
                self.logger.info(f"Placed buy order at {grid_prices[i]}")
                
                # Place sell order
                sell_order = self.client.create_order(
                    symbol=self.symbol,
                    side=SIDE_SELL,
                    type=ORDER_TYPE_LIMIT,
                    timeInForce=TIME_IN_FORCE_GTC,
                    quantity=quantity,
                    price=str(grid_prices[i+1])
                )
                self.active_orders[sell_order['orderId']] = sell_order
                self.logger.info(f"Placed sell order at {grid_prices[i+1]}")
                
                time.sleep(0.1)  # Avoid API rate limits
                
            except Exception as e:
                self.logger.error(f"Error placing grid orders: {e}")
    
    def monitor_and_replace_orders(self):
        """Monitor filled orders and replace them"""
        while True:
            try:
                # Check for copy trading opportunities
                self.copy_trader_orders()
                
                # Monitor existing orders
                for order_id in list(self.active_orders.keys()):
                    order = self.client.get_order(
                        symbol=self.symbol,
                        orderId=order_id
                    )
                    
                    if order['status'] == 'FILLED':
                        self.logger.info(f"Order {order_id} filled at {order['price']}")
                        del self.active_orders[order_id]
                        
                        # Calculate new order price
                        new_price = float(order['price'])
                        if order['side'] == SIDE_BUY:
                            new_price *= (1 + self.grid_size_percent/100)
                        else:
                            new_price *= (1 - self.grid_size_percent/100)
                        
                        # Place new order
                        quantity = self.calculate_quantity(new_price)
                        if quantity:
                            new_order = self.client.create_order(
                                symbol=self.symbol,
                                side=SIDE_SELL if order['side'] == SIDE_BUY else SIDE_BUY,
                                type=ORDER_TYPE_LIMIT,
                                timeInForce=TIME_IN_FORCE_GTC,
                                quantity=quantity,
                                price=str(new_price)
                            )
                            self.active_orders[new_order['orderId']] = new_order
                            self.logger.info(f"Placed replacement order at {new_price}")
                
                time.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Error in monitor_and_replace_orders: {e}")
                time.sleep(1)
    
    def run(self):
        """Main method to run the enhanced grid trading bot"""
        self.logger.info("Starting Enhanced Grid Trading Bot")
        try:
            # Setup initial grid
            grid_prices = self.setup_grid()
            if not grid_prices:
                self.logger.error("Failed to setup grid prices")
                return
                
            self.logger.info(f"Created grid with {len(grid_prices)} price levels")
            
            # Place initial orders
            self.place_grid_orders(grid_prices)
            self.logger.info("Placed initial grid orders")
            
            # Monitor and replace orders
            self.monitor_and_replace_orders()
            
        except Exception as e:
            self.logger.error(f"Error running bot: {e}")

if __name__ == "__main__":
    # Example usage
    bot = EnhancedGridBot(
        symbol='BTCUSDT',
        grid_levels=10,
        grid_size_percent=1.0,
        copy_trader_id=None  # Add trader ID to copy
    )