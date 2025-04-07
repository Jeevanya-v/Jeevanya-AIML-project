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
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# Select relevant columns based on dataset structure
required_columns = ["location", "carpet area", "price (in rupees)", "status", "ownership"]
df = df[required_columns].dropna()

# Convert 'carpet area' to numeric, handling different units (sqft & sqm)
def convert_area(value):
    value = str(value).lower().replace(',', '').strip()
    if 'sqft' in value:
        return float(value.replace('sqft', '').strip())
    elif 'sqm' in value:
        return float(value.replace('sqm', '').strip()) * 10.764  # Convert sqm to sqft
    else:
        try:
            return float(value)
        except ValueError:
            return np.nan

df["carpet area"] = df["carpet area"].apply(convert_area)

# Handle missing values
imputer_numeric = SimpleImputer(strategy="median")
df["carpet area"] = imputer_numeric.fit_transform(df[["carpet area"]])

# Encode categorical variables
label_encoders = {}
for col in ["location", "status", "ownership"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Scale numerical data
scaler = StandardScaler()
df[["carpet area"]] = scaler.fit_transform(df[["carpet area"]])

# Define input (X) and target (y)
X = df[["location", "carpet area", "status", "ownership"]]
y = df["price (in rupees)"]

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import numpy as np
from sklearn.model_selection import GridSearchCV

# Hyperparameter tuning using Grid Search
param_grid = {
    'n_estimators': [500, 1000],  # More trees for better learning
    'learning_rate': [0.01, 0.05],  # Lower learning rate improves stability
    'max_depth': [7, 10],  # Allow deeper trees for complex patterns
    'num_leaves': [31, 50],  # More leaves allow better split decisions
    'min_child_samples': [10, 20]  # Controls minimum samples per leaf
}

lgb_model = lgb.LGBMRegressor(random_state=42)
grid_search = GridSearchCV(lgb_model, param_grid, cv=5, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model after tuning
best_lgb_model = grid_search.best_estimator_

# Predict and evaluate
y_pred = best_lgb_model.predict(X_test)
mae_lgb = mean_absolute_error(y_test, y_pred)
mse_lgb = mean_squared_error(y_test, y_pred)
rmse_lgb = np.sqrt(mse_lgb)
r2_lgb = r2_score(y_test, y_pred)  # R² should now be positive!

# Print evaluation metrics
print(f"MAE: {mae_lgb:.2f}, MSE: {mse_lgb:.2f}, RMSE: {rmse_lgb:.2f}, R² Score: {r2_lgb:.4f}")

# Save the best trained model
joblib.dump(best_lgb_model, "best_lgb_model.pkl")
def safe_encode_location(location, label_encoders):
    """ Encode location safely, returning most common value if unknown """
    if location in label_encoders["location"].classes_:
        return label_encoders["location"].transform([location])[0]
    else:
        return np.argmax(np.bincount(df["location"]))  # Assign most common location
def predict_future_price(location, carpet_area, status, ownership, years=5):
    """ Predict land price in the future using trained LightGBM model """
    loc_encoded = safe_encode_location(location, label_encoders)
    
    # Create DataFrame to maintain feature names
    input_features = pd.DataFrame([[loc_encoded, carpet_area, status, ownership]], 
                                  columns=X.columns)
    
    # Scale the carpet area
    input_features["carpet area"] = scaler.transform(input_features[["carpet area"]])

    # Predict current price
    current_price = lgb_model.predict(input_features)[0]
    
    # Assume an annual price growth rate
    avg_growth_rate = 0.05  
    future_price = current_price * ((1 + avg_growth_rate) ** years)
    
    return future_price


# Example: Get user input 
location_input = input("Enter location: ")
carpet_area_input = float(input("Enter carpet area (in sqft): "))
status_input = int(input("Enter status (0 for Under Construction, 1 for Ready to Move): "))
ownership_input = int(input("Enter ownership type (0 for Leasehold, 1 for Freehold): "))
years_input = int(input("Enter number of years to predict: "))

# Predict price
future_price = predict_future_price(location_input, carpet_area_input, status_input, ownership_input, years_input)
print(f'Predicted price after {years_input} years: {future_price:.2f} INR')
