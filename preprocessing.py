import pandas as pd
import numpy as np

# Load raw data
df = pd.read_csv('sample.csv')

# Fill ID
df['Participant_ID'] = df['Participant_ID'].ffill().bfill()

# Fill CESD scores
df['K_CESD_R'] = df['K_CESD_R'].ffill().bfill()

# Remove non-stimuli images
df = df[~df['STIMULI_NO'].isin(['BLACK', 'WHITE', 'GREY'])]

# Define outlier marking function
def mark_outliers_pupil(pupil):
    for i in range(1, len(pupil)):
        if abs(pupil.iloc[i] - pupil.iloc[i-1]) / pupil.iloc[i-1] > 0.1:
            pupil.iloc[i] = np.nan
            ref_value = pupil.iloc[i-1]
            for j in range(i+1, len(pupil)):
                if abs(pupil.iloc[j] - ref_value) / ref_value <= 0.05:
                    break
                pupil.iloc[j] = np.nan
    return pupil

# Define preprocessing function
def preprocess(group):
    pupil = group['D_RATIO'].copy()
    pupil = mark_outliers_pupil(pupil)

    if pupil.count() >= 2:
        pupil = pupil.interpolate(method='pchip', limit=30, limit_direction='both')
    else:
        return None

    pupil[(pupil < 40) | (pupil > 160)] = np.nan

    if pupil.count() >= 2:
        pupil = pupil.interpolate(method='pchip', limit=30, limit_direction='both')
    else:
        return None

    if pupil.isna().any():
        return None

    group['D_RATIO'] = pupil
    return group

# Apply preprocessing
processed_samples = []
for name, group in df.groupby(['Participant_ID', 'STIMULI_NO']):
    processed_group = preprocess(group)
    if processed_group is not None:
        processed_samples.append(processed_group)

df_cleaned = pd.concat(processed_samples)

# Final clipping to 40~160
df_cleaned = df_cleaned[(df_cleaned['D_RATIO'] >= 40) & (df_cleaned['D_RATIO'] <= 160)]

# Save to CSV
df_cleaned.to_csv('cleaned.csv', index=False)
print("saved: cleaned.csv")
