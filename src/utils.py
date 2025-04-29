import ta
import math
import pandas as pd
import numpy as np

EMA_1 = 60
EMA_2 = 120
MACD_SLOW_WINDOW = 120
MACD_FAST_WINDOW = 45
MACD_SIGNAL_WINDOW = 9
STOCH_WINDOW = 45

def add_indicators(df):
    """
    Add the following indicators to the pandas dataframe:
    EMA_SLOW with window  120
    EMA_FAST with window 60
    MACD with slow window 120, fast window 45 and signal window 9
    Stochastic with window 45

    :param df:
    :return:
    """
    df['ema_slow'] = ta.trend.EMAIndicator(df['close'], window=EMA_2).ema_indicator()
    df['ema_fast'] = ta.trend.EMAIndicator(df['close'], window=EMA_1).ema_indicator()

    # MACD
    macd = ta.trend.MACD(df['close'], window_slow=MACD_SLOW_WINDOW, window_fast=MACD_FAST_WINDOW,
                         window_sign=MACD_SIGNAL_WINDOW)
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    df['MACD_diff'] = macd.macd_diff()

    # Stochastic
    stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'],
                                             window=STOCH_WINDOW)
    df['stoch'] = stoch.stoch()
    df['stoch_signal'] = stoch.stoch_signal()

    return df


def calculate_batch_size(total_samples, target_batch_count=100):
    """Calculates the batch size as the nearest power of 2 for 100 batches.

    Args:
        total_samples: The total number of samples.

    Returns:
        The calculated batch size.
    """

    ideal_batch_size = total_samples // target_batch_count
    power_of_two = round(math.log2(ideal_batch_size))
    batch_size = 2**power_of_two
    return batch_size

def get_all_data(data_base: str):
    data = pd.read_csv(f"{data_base}/NIFTY 50_minute_data.csv", parse_dates=['date'])
    data['cal_date'] = pd.to_datetime(data['date'].dt.date)

    daily_data = pd.read_csv(f"{data_base}/NIFTY 50_daily_data.csv", parse_dates=['date'])
    daily_data['date'] = pd.to_datetime(daily_data['date'])
    # Calculate the percentage gap between the current day's open and the previous day's close
    daily_data['previous_close'] = daily_data['close'].shift(1)
    daily_data['gap_percentage'] = ((daily_data['open'] - daily_data['previous_close']) / daily_data[
        'previous_close']) * 100

    gap_threshold = 0.7
    # Categorize the gap as 'gap_up', 'gap_down', or 'no_gap'
    daily_data['gap_type'] = np.where(daily_data['gap_percentage'] > gap_threshold, 'gap_up',
                                      np.where(daily_data['gap_percentage'] < -gap_threshold, 'gap_down', 'no_gap'))

    vix_min = pd.read_csv(f"{data_base}/INDIA VIX_minute.csv", parse_dates=['date'])
    # Rename columns in the 'data' DataFrame
    vix_min = vix_min.rename(columns={
        'date': 'vix_date',
        'open': 'vix_open',
        'high': 'vix_high',
        'low': 'vix_low',
        'close': 'vix_close',
        'volume': 'vix_volume'
    })
    vix_min.drop(columns=['vix_volume'], inplace=True)

    # Rename columns in the 'data' DataFrame
    data = data.rename(columns={
        'date': 'min_date',
        'open': 'min_open',
        'high': 'min_high',
        'low': 'min_low',
        'close': 'min_close',
        'volume': 'min_volume'
    })
    data['hour'] = data['min_date'].dt.hour
    data['hour'] = data['hour'].astype(str)

    data["hour_start"] = data["min_date"].dt.floor("h")
    # Calculate the return from the start of the hour
    data = data.merge(data.groupby("hour_start")["min_close"].first(), on="hour_start", suffixes=("", "_start"))
    data["hourly_return"] = (data["min_close"] - data["min_close_start"]) / data["min_close_start"]
    merged_data = pd.merge(data, daily_data, left_on='cal_date', right_on='date', how='left')
    merged_data = pd.merge(merged_data, vix_min, left_on='min_date', right_on='vix_date', how='left')
    merged_data = merged_data.drop(['volume', 'min_volume', 'previous_close', 'open', 'high', 'low',
                                    'close', 'date', 'cal_date', 'vix_date', 'hour_start'], axis=1)
    merged_data = merged_data.rename(columns={
        'min_open': 'open',
        'min_high': 'high',
        'min_low': 'low',
        'min_close': 'close',
    })
    return merged_data