import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# ==============================
# 1. Load Dataset
# ==============================

df = pd.read_csv("data/student_data.csv")

X = df.drop("Placement_Status", axis=1)
y = df["Placement_Status"]


# ==============================
# 2. Feature Types
# ==============================

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


# ==============================
# 3. Preprocessing
# ==============================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numerical",
            StandardScaler(),
            numerical_features
        )
    ]
)


# ==============================
# 4. Models
# ==============================

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        random_state=42,
        class_weight="balanced"
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
}


# ==============================
# 5. Train/Test Split
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==============================
# 6. Train & Evaluate
# ==============================

results = []

print("=" * 70)
print("STUDENT PLACEMENT - MODEL COMPARISON")
print("=" * 70)

for name, model in models.items():

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test,
        y_pred,
        pos_label="Placed"
    )
    recall = recall_score(
        y_test,
        y_pred,
        pos_label="Placed"
    )
    f1 = f1_score(
        y_test,
        y_pred,
        pos_label="Placed"
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1
    })


# ==============================
# 7. Results
# ==============================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="F1_Score",
    ascending=False
)

print("\nModel Performance:\n")

print(
    results_df.to_string(
        index=False,
        formatters={
            "Accuracy": "{:.2%}".format,
            "Precision": "{:.2%}".format,
            "Recall": "{:.2%}".format,
            "F1_Score": "{:.2%}".format
        }
    )
)


# ==============================
# 8. Best Model
# ==============================

best_model = results_df.iloc[0]

print("\n" + "=" * 70)
print("BEST MODEL")
print("=" * 70)

print(f"Model     : {best_model['Model']}")
print(f"Accuracy  : {best_model['Accuracy']:.2%}")
print(f"Precision : {best_model['Precision']:.2%}")
print(f"Recall    : {best_model['Recall']:.2%}")
print(f"F1 Score  : {best_model['F1_Score']:.2%}")

print("=" * 70)