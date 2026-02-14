# src/train_model.py

# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Load dataset
df = pd.read_csv("data/library.csv")

# Show dataset columns (for debugging)
print("Columns in dataset:", df.columns)

# Feature engineering
# Create book_age from PublishYear
df["book_age"] = 2024 - df["PublishYear"]

# Select useful features
features = ["pagesNumber", "CountsOfReview", "book_age"]
target = "Rating"

# Remove missing values
df = df.dropna(subset=features + [target])

X = df[features]
y = df[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model using joblib
joblib.dump(model, "model/library_model.pkl")

print("Model saved successfully!")

