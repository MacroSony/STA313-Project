import numpy as np
import pandas as pd

def weighted_quantile(values, quantiles, sample_weight=None):
    """ Call this function to get weighted percentiles. """
    values = np.array(values)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)
    
    sorter = np.argsort(values)
    values = values[sorter]
    sample_weight = sample_weight[sorter]
    
    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    weighted_quantiles /= np.sum(sample_weight)
    
    return np.interp(quantiles, weighted_quantiles, values)

# Load your data (assuming CSV for this example)
df = pd.read_csv("census_full.csv")

# ---------------------------------------------------------
# STEP 1: INFLATION ADJUSTMENT (Target: 2021 Dollars)
# ---------------------------------------------------------
# You need to fill these with the actual CPI multipliers from Bank of Canada
# Formula: (2021 CPI / Year CPI)
cpi_multipliers = {
    1991: 1.76, 
    1996: 1.58,
    2001: 1.45,
    2006: 1.31,
    2011: 1.19,
    2016: 1.10,
    2021: 1.00
}

# Map the multiplier to the Year column
df['cpi_factor'] = df['YEAR'].map(cpi_multipliers)

# Calculate Real Wages
# We assume 'Wages' is the column name. Handle missing/zero wages first.
df = df[(df['Wages'] > 0) & (df['Wages'].notna())]
df['Real_Wages'] = df['Wages'] * df['cpi_factor']

# ---------------------------------------------------------
# STEP 2: BASIC CLEANING
# ---------------------------------------------------------
# Filter for Age 15+
# df = df[df['AGEGRP'] >= 15] # Ensure your AGEGRP column is numeric

# Standardize Immigrant Status for cleaner pivoting later
df['IMMSTAT_CLEAN'] = np.where(
    df['IMMSTAT'] == 'Immigrant', 'Immigrant', 'Non-Immigrant'
)

# ---------------------------------------------------------
# STEP 3: THE GRAND AGGREGATION (The "Cube")
# ---------------------------------------------------------
# We define a custom aggregation function to handle weights
def weighted_stats(x):
    # x is a DataFrame of the specific group
    w = x['WEIGHT']
    v = x['Real_Wages']
    
    # Calculate stats
    try:
        avg = np.average(v, weights=w)
        # Get quantiles for box plot (25th, 50th, 75th, 95th)
        q25, q50, q75, q95 = weighted_quantile(v, [0.25, 0.50, 0.75, 0.95], sample_weight=w)
    except:
        avg, q25, q50, q75, q95 = 0, 0, 0, 0, 0
        
    return pd.Series({
        'Mean_Wage': avg,
        'Population': w.sum(),
        'Q1': q25,
        'Median': q50,
        'Q3': q75,
        'P95': q95 # Cap outliers at 95th percentile
    })

# Group by EVERY dimension used in your dashboard filters
# This might take a minute to run, but it only needs to happen once.
group_cols = [
    'YEAR', 'PR', 'CMA', 'Design_Education', 
    'GENSTAT', 'IMMCAT5', 'IMMSTAT_CLEAN'
]

# Apply the weighted stats
df_agg = df.groupby(group_cols).apply(weighted_stats).reset_index()

# ---------------------------------------------------------
# STEP 4: CLEAN UP NOISE
# ---------------------------------------------------------
# Privacy/Robustness check: Remove groups with very few people
df_agg = df_agg[df_agg['Population'] > 30]

# Round numbers for smaller file size
df_agg['Mean_Wage'] = df_agg['Mean_Wage'].round(0)
df_agg['Median'] = df_agg['Median'].round(0)

# Export this file -> This powers your Line Charts and Side Panel
df_agg.to_csv("processed_dashboard_data.csv", index=False)
print("Aggregation Complete.")