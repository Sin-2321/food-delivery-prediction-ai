import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle

print("⏳ Generating realistic food delivery dataset...")
np.random.seed(42)
n_samples = 5000

# Generating realistic numbers for training
distance = np.random.uniform(1.0, 12.0, n_samples)      # Kilometers
num_items = np.random.randint(1, 8, n_samples)          # Number of dishes ordered
hour_of_day = np.random.randint(11, 23, n_samples)      # 11 AM to 11 PM
weather_severity = np.random.randint(1, 6, n_samples)   # 1 (Clear) to 5 (Heavy Rain)
traffic_density = np.random.randint(1, 6, n_samples)    # 1 (Low) to 5 (Jam)

# Base delivery time is 15 minutes + real-world delays
base_time = 15 
delivery_time = (
    base_time 
    + (distance * 2.5) 
    + (num_items * 1.8) 
    + (weather_severity * 4.5) 
    + (traffic_density * 5.0)
    + np.random.normal(0, 3, n_samples) # Adding random human noise/delays
)

# Create DataFrame
df = pd.DataFrame({
    'distance_km': distance,
    'num_items': num_items,
    'hour': hour_of_day,
    'weather_severity': weather_severity,
    'traffic_density': traffic_density,
    'delivery_time_mins': delivery_time
})

# Split into inputs (X) and target output (y)
X = df.drop(columns=['delivery_time_mins'])
y = df['delivery_time_mins']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("🤖 Training Random Forest Regression Model...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Calculate Accuracy Score
score = model.score(X_test, y_test)
print(f"✅ Model Training Complete! Accuracy: {score * 100:.2f}%")

# Save the trained model to a file
with open('delivery_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("💾 Saved engine as 'delivery_model.pkl'")