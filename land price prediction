import pandas as pd

# Load both datasets


df1 = pd.read_csv(r"C:\Users\V.R.JEEVANYA\Downloads\dataset\house_prices.csv")
df2 = pd.read_csv(r"C:\Users\V.R.JEEVANYA\Downloads\dataset\Real Estate Data V211.csv")

# Standardize column names
df1.columns = df1.columns.str.strip().str.lower()
df2.columns = df2.columns.str.strip().str.lower()

# Merge datasets if they have common columns, else concatenate them
if any(col in df1.columns for col in df2.columns):
    df = pd.merge(df1, df2, how='outer')
else:
    df = pd.concat([df1, df2], axis=0)

# Display the first few rows
df.head()
