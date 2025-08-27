from ta import momentum, trend, volatility
import pandas as pd
import argparse
from src.utils.helpers import write_df_to_csv

def add_ta_features(df):
    """Add technical analysis features to the DataFrame."""
    df['rsi'] = momentum.RSIIndicator(df['close']).rsi()
    macd = trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['sma_20'] = trend.SMAIndicator(df['close'], window=20).sma_indicator()
    df['ema_20'] = trend.EMAIndicator(df['close'], window=20).ema_indicator()
    bb = volatility.BollingerBands(df['close'])
    df['bb_bbm'] = bb.bollinger_mavg()
    df['bb_bbh'] = bb.bollinger_hband()
    df['bb_bbl'] = bb.bollinger_lband()
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add technical analysis features to stock data.")
    parser.add_argument('--input_file_y2', type=str, required=True, help="Input CSV file with historical stock data")
    parser.add_argument('--input_file_new', type=str, required=True, help="Input CSV file with recent stock data")
    parser.add_argument('--raw_dir', type=str, required=True, help="Directory for raw data files")
    parser.add_argument('--processed_dir', type=str, required=True, help="Directory for processed data files")

    args = parser.parse_args()

    hist_df = pd.read_csv(f'{args.raw_dir}/{args.input_file_y2}')
    hist_df = add_ta_features(hist_df)

    new_df = pd.read_csv(f'{args.raw_dir}/{args.input_file_new}')
    new_df = add_ta_features(new_df)

    # Write processed feature files (restored)
    write_df_to_csv(hist_df, args.processed_dir, args.input_file_y2)
    write_df_to_csv(new_df, args.processed_dir, args.input_file_new)