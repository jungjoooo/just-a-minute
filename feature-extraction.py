import pandas as pd
import numpy as np
from scipy.integrate import trapezoid

# Data Loading
df = pd.read_csv('cleaned.csv')

# Standardize
def standardize(series):
    return (series / series.iloc[0]) * 100

# Extract
def extract_features(group):
    pupil = group['D_RATIO'].reset_index(drop=True)
    features = {}

    features['max'] = pupil.max()
    features['min'] = pupil.min()
    features['mean'] = pupil.mean()
    features['std'] = pupil.std()
    features['start'] = pupil.iloc[0]
    features['finish'] = pupil.iloc[-1]
    
    # Peak after frame 50
    peak = pupil.iloc[50:].max()
    peak_idx = pupil.iloc[50:].idxmax()
    features['peak'] = peak
    features['peak_ratio'] = peak / pupil.min()
    features['peak_latency'] = peak_idx

    # Drop latency and drop ratio
    min_idx = pupil.idxmin()
    features['drop_latency'] = min_idx
    features['drop_ratio'] = pupil.min() / pupil.iloc[0]

    # Recovery ratio
    features['recovery_ratio'] = pupil.iloc[-1] / pupil.iloc[0]

    # Response duration
    features['response_duration'] = features['peak_latency'] - features['drop_latency']

    # Reaction extent
    features['reaction_extent'] = features['drop_ratio'] + features['recovery_ratio']

    # Stability
    features['stability'] = features['mean'] / features['std']

    # Range
    features['range'] = features['max'] - features['min']

    # Baseline deviation
    features['baseline_deviation'] = features['start'] - features['mean']

    # Area under curve
    features['AUC'] = trapezoid(pupil)

    # Change sum
    features['change_sum'] = np.sum(np.abs(np.diff(pupil)))

    # Standardized pupil data
    pupil_std = standardize(pupil)

    # Standardized change sum
    features['s_change_sum'] = np.sum(np.abs(np.diff(pupil_std)))

    # Standardized AUC
    features['s_AUC'] = trapezoid(pupil_std)

    # 60frame cut
    for i in range(6):
        start_frame = i * 60
        end_frame = (i + 1) * 60

        segment = pupil.iloc[start_frame:end_frame]
        segment_std = pupil_std.iloc[start_frame:end_frame]

        features[f'AUC{i+1}'] = trapezoid(segment)
        features[f'change_sum{i+1}'] = np.sum(np.abs(np.diff(segment)))
        features[f's_change_sum{i+1}'] = np.sum(np.abs(np.diff(segment_std)))
        features[f's_AUC{i+1}'] = trapezoid(segment_std)

    # K_CESD_R labels
    features['K_CESD_R'] = group['K_CESD_R'].iloc[0]
    features['depression_label'] = 1 if group['K_CESD_R'].iloc[0] >= 3.0 else 0

    return pd.Series(features)

# Group feature extract
features_df = df.groupby(['Participant_ID', 'STIMULI_NO']).apply(extract_features).reset_index()

# Drop NA
features_df.dropna(inplace=True)

# Save
features_df.to_csv('data.csv', index=False)

print("saved: data.csv")
