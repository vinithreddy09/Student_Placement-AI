import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
df = pd.read_csv("data/student_data.csv")

# Features and target
X = df.drop("Placement_Status", axis=1)
y = df["Placement_Status"]

# Categorical columns
categorical_features = ["Gender"]

# Numerical columns
numerical_features = [
    "Age",
    "CGPA",
    "Tenth_Percentage",
    "Twelfth_Percentage",
    "Aptitude_Score",
    "Coding_Score",
    "Communication_Score",
    "Technical_Skills",
    "Certifications",
    "Internship_Experience",
    "Projects"
]

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ],
    remainder="passthrough"
)

# Random Forest model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

# Complete pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Train
pipeline.fit(X_train, y_train)

# Predictions
y_pred = pipeline.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("=" * 50)
print("STUDENT PLACEMENT PREDICTION MODEL")
print("=" * 50)

print(f"\nAccuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save model
joblib.dump(pipeline, "models/placement_model.pkl")

print("\nModel saved successfully!")
print("Location: models/placement_model.pkl")