import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Load dataset
df = pd.read_csv("data/student_data.csv")

X = df.drop("Placement_Status", axis=1)
y = df["Placement_Status"]


# Categorical and numerical columns
categorical_features = ["Gender"]

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
        (
            "num",
            StandardScaler(),
            numerical_features
        ),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


models = {

    "Logistic Regression": Pipeline([
        ("preprocessor", preprocessor),
        (
            "model",
            LogisticRegression(
                max_iter=2000
            )
        )
    ]),

    "Random Forest": Pipeline([
        ("preprocessor", preprocessor),
        (
            "model",
            RandomForestClassifier(
                random_state=42,
                class_weight="balanced"
            )
        )
    ]),

    "Gradient Boosting": Pipeline([
        ("preprocessor", preprocessor),
        (
            "model",
            GradientBoostingClassifier(
                random_state=42
            )
        )
    ])
}


# Hyperparameter grids

parameters = {

    "Logistic Regression": {
        "model__C": [0.01, 0.1, 1, 10, 100]
    },

    "Random Forest": {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [None, 5, 10, 15],
        "model__min_samples_split": [2, 5, 10]
    },

    "Gradient Boosting": {
        "model__n_estimators": [50, 100, 150],
        "model__learning_rate": [0.01, 0.05, 0.1],
        "model__max_depth": [2, 3, 4]
    }
}


results = []

best_model = None
best_f1 = 0
best_name = ""


print("\nMODEL TUNING STARTED")
print("=" * 60)


for name in models:

    print(f"\nTraining: {name}")

    grid = GridSearchCV(
        models[name],
        parameters[name],
        cv=5,
        scoring="f1",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    prediction = grid.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        prediction
    )

    precision = precision_score(
        y_test,
        prediction,
        pos_label="Placed"
    )

    recall = recall_score(
        y_test,
        prediction,
        pos_label="Placed"
    )

    f1 = f1_score(
        y_test,
        prediction,
        pos_label="Placed"
    )

    results.append([
        name,
        accuracy,
        precision,
        recall,
        f1
    ])

    print("Best Parameters:")
    print(grid.best_params_)

    print(f"Accuracy : {accuracy:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall   : {recall:.2%}")
    print(f"F1 Score : {f1:.2%}")

    if f1 > best_f1:

        best_f1 = f1
        best_model = grid.best_estimator_
        best_name = name


# Save best model

import joblib

joblib.dump(
    best_model,
    "models/placement_model.pkl"
)


print("\n" + "=" * 60)

print("BEST MODEL")
print("=" * 60)

print(f"Model     : {best_name}")
print(f"F1 Score  : {best_f1:.2%}")

print("\nBest model saved to:")
print("models/placement_model.pkl")

print("\nMODEL TUNING COMPLETED!")