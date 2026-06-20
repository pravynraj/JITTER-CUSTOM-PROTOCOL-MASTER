import pandas as pd
import numpy as np
import argparse

def process_log(input_file, output_x, output_y, window_size=20, horizon=15):
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    
    expected_cols = ['time', 'seq', 'rtt', 'delay', 'jitter', 'loss']
    for col in expected_cols:
        if col not in df.columns:
            raise ValueError(f"Missing expected column: {col}")
            
    # Sort by time and sequence just to be safe
    df = df.sort_values(by=['time', 'seq']).reset_index(drop=True)
    
    # Handle missing values for lost packets
    # Ffill maintains the last valid rtt and delay. 
    # For jitter, substitute 0 because no new delay gap is observed.
    if hasattr(df['rtt'], 'ffill'):
        df['rtt'] = df['rtt'].ffill().fillna(0)
        df['delay'] = df['delay'].ffill().fillna(0)
    else:
        df['rtt'] = df['rtt'].fillna(method='ffill').fillna(0)
        df['delay'] = df['delay'].fillna(method='ffill').fillna(0)
    
    df['jitter'] = df['jitter'].fillna(0)
    
    # Select our feature columns
    features = ['time', 'rtt', 'delay', 'jitter', 'loss']
    df_features = df[features].copy()
    
    # Base time at 0 for cleaner normalization ranges
    df_features['time'] = df_features['time'] - df_features['time'].min()
    
    print("Normalizing features...")
    feature_mins = df_features.min()
    feature_maxs = df_features.max()
    
    ptp = feature_maxs - feature_mins
    ptp = ptp.replace(0, 1) # Avoid division by zero
    
    df_norm = (df_features - feature_mins) / ptp
    
    data = df_norm.values
    raw_jitter = df['jitter'].values
    
    X = []
    y = []
    
    print(f"Generating sequences (window_size={window_size}, horizon={horizon})...")
    # Generate sliding windows
    for i in range(len(data) - window_size - horizon + 1):
        # Get sequential block of size 'window_size'
        seq_x = data[i:i+window_size]
        
        # View next 'horizon' packets to check for jitter values
        future_jitter = raw_jitter[i+window_size : i+window_size+horizon]
        
        # Compute dynamic threshold based on the current window
        window_jitter = raw_jitter[i:i+window_size]
        mean_jitter = np.mean(window_jitter)
        std_jitter = np.std(window_jitter)
        
        # Avoid issues if variance is zero
        if std_jitter == 0:
            std_jitter = 1e-6
            
        # Lower multiplier (0.5) makes the detector more sensitive
        dynamic_threshold = mean_jitter + 0.5 * std_jitter
        
        # Use the max future jitter to catch the worst spike in the horizon
        future_max_jitter = np.max(future_jitter)
        label = 1 if future_max_jitter > dynamic_threshold else 0
        
        X.append(seq_x)
        y.append(label)
        
    X = np.array(X)
    y = np.array(y)
    
    print(f"Saving to {output_x} and {output_y}...")
    np.save(output_x, X)
    np.save(output_y, y)
    
    print("Processing complete!")
    print(f"Dataset shape - X: {X.shape}, y: {y.shape}")
    
    # Print class balance as requested
    unique, counts = np.unique(y, return_counts=True)
    class_counts = dict(zip(unique, counts))
    count_0 = class_counts.get(0, 0)
    count_1 = class_counts.get(1, 0)
    
    print(f"Label Distribution -> Label 0: {count_0}, Label 1: {count_1}")
    print(f"Percentage of positive labels (spikes in horizon): {np.mean(y)*100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process UDP network log into LSTM dataset")
    parser.add_argument("--input", "-i", type=str, default="sender_log.csv", help="Input log CSV file")
    parser.add_argument("--out_x", "-x", type=str, default="X_dataset.npy", help="Output numpy array for features")
    parser.add_argument("--out_y", "-y", type=str, default="y_dataset.npy", help="Output numpy array for labels")
    parser.add_argument("--window", "-w", type=int, default=20, help="Sliding window size (past packets)")
    parser.add_argument("--horizon", "-r", type=int, default=15, help="Prediction horizon (future packets)")
    args = parser.parse_args()
    
    try:
        process_log(args.input, args.out_x, args.out_y, 
                    window_size=args.window, 
                    horizon=args.horizon)
    except FileNotFoundError:
        print(f"Error: File '{args.input}' not found. Please check paths.")
    except Exception as e:
        print(f"An error occurred: {e}")
