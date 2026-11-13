import pandas as pd
df = pd.read_csv("master_features_scaled_v2.csv")

    # Compare the 95th percentile and maximums rather than just means
print(df.groupby('regime_name')[['rms_dyn_g', 'p2p_dyn_g', 'gyro_rms_dps']].agg(['mean', 'max', lambda x: x.quantile(0.95)]))