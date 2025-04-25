import numpy as np
import pandas as pd
import random

from utils.classify_window import classify_window
#Trading Strategies

def strategy_head_and_shoulders(forecast):
    """Sell at the peak of the right shoulder, buy after a confirmation drop."""
    if len(forecast) < 4:
        return {"buy": [], "sell": []}

    peak_idx = int(np.argmax(forecast))
    buy_idx = peak_idx + 3 if (peak_idx + 3) < len(forecast) else len(forecast) - 1

    return {"buy": [buy_idx], "sell": [peak_idx]}

def strategy_double_bottom(forecast):
    """Buy after the second dip, sell after recovery."""
    forecast = np.array(forecast).flatten()

    if len(forecast) < 2:
        return {"buy": [], "sell": []}

    try:
        bottom_indices = np.argsort(forecast)[:2]
        bottom_indices = sorted(int(i) for i in bottom_indices)
    except Exception as e:
        print("Error in strategy_double_bottom - forecast:", forecast)
        raise e

    second_dip = bottom_indices[1]
    sell_idx = second_dip + 5 if (second_dip + 5) < len(forecast) else len(forecast) - 1

    return {"buy": [second_dip], "sell": [sell_idx]}

def strategy_ascending_triangle(forecast):
    """Buy mid-way into the pattern, sell near the predicted resistance."""
    if len(forecast) < 2:
        return {"buy": [], "sell": []}

    mid_idx = len(forecast) // 2
    sell_idx = len(forecast) - 1

    return {"buy": [mid_idx], "sell": [sell_idx]}

def strategy_none(_forecast):
    return {"buy": [], "sell": []}


# === Trader Class ===

class Trader:
    def __init__(self):
        self.strategies = {
            "head_and_shoulders": strategy_head_and_shoulders,
            "double_bottom": strategy_double_bottom,
            "ascending_triangle": strategy_ascending_triangle,
            "none": strategy_none
        }

    def get_signals(self, forecast, pattern_label):
        strategy = self.strategies.get(pattern_label.lower(), strategy_none)
        # return strategy(np.array(forecast))  # Ensure it's passed as a numpy array
        forecast_arr = np.array(forecast).flatten()  # Ensures 1D array
        return strategy(forecast_arr)

# === Market Simulation ===
class MarketSimulator:
    def __init__(self, processed_data, initial_budget=100000, price_tolerance=0.5):
        """
        processed_data: list of dicts, each containing forecast_window, forecasts, labels
        """
        self.processed_data = processed_data
        self.trader = Trader()
        self.initial_budget = initial_budget
        self.budget = initial_budget
        self.position = 0
        self.trade_log = []
        self.price_tolerance = price_tolerance

    def simulate(self):
        for idx, entry in enumerate(self.processed_data):
            try:
                forecast = np.array(entry['forecast'], dtype=float).flatten()
                window = np.array(entry['window'], dtype=float).flatten()
                label = entry['label']
            except Exception as e:
                print(f"[ERROR] Failed to parse entry {idx} due to type error: {e}")
                continue
            # print(forecast, window, label)
            signals = self.trader.get_signals(forecast, label)

            # --- Buy Trades ---
            for buy_idx in signals['buy']:
                if buy_idx < len(forecast):
                    target_price = forecast[buy_idx]
                    # print(f"BUY: {target_price}-{window}")
                    if np.any(np.abs(window - target_price) <= self.price_tolerance):
                        if self.budget >= target_price:
                            self.budget -= target_price
                            self.position += 1
                            self.trade_log.append((idx, "buy", target_price, self.budget))

            # --- Sell Trades ---
            for sell_idx in signals['sell']:
                if sell_idx < len(forecast):
                    target_price = forecast[sell_idx]
                    # print(f"BUY: {buy_idx}-{len(forecast)}-{label}-{target_price}-{np.abs(window - target_price)}")
                    if np.any(np.abs(window - target_price) <= self.price_tolerance):
                        if self.position > 0:
                            self.budget += target_price
                            self.position -= 1
                            self.trade_log.append((idx, "sell", target_price, self.budget))

            if self.budget <= 0:
                break

        return self.trade_log
