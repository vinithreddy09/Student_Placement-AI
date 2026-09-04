import pandas as pd
import numpy as np

np.random.seed(42)

n = 500

data = {
    "Age": np.random.randint(20, 24, n),
    "Gender": np.random.choice(["Male", "Female"], n),
    "CGPA": np.round(np.random.uniform(5.5, 9.8, n), 2),
    "Tenth_Percentage": np.round(np.random.uniform(55, 98, n), 2),
    "Twelfth_Percentage": np.round(np.random.uniform(55, 98, n), 2),
    "Aptitude_Score": np.random.randint(30, 101, n),
    "Coding_Score": np.random.randint(20, 101, n),
    "Communication_Score": np.random.randint(30, 101, n),
    "Technical_Skills": np.random.randint(1, 11, n),
    "Certifications": np.random.randint(0, 6, n),
    "Internship_Experience": np.random.randint(0, 3, n),
    "Projects": np.random.randint(0, 6, n)
}

df = pd.DataFrame(data)

# Calculate a realistic placement score
score = (
    df["CGPA"] * 8
    + df["Tenth_Percentage"] * 0.10
    + df["Twelfth_Percentage"] * 0.10
    + df["Aptitude_Score"] * 0.18
    + df["Coding_Score"] * 0.22
    + df["Communication_Score"] * 0.12
    + df["Technical_Skills"] * 2
    + df["Certifications"] * 2
    + df["Internship_Experience"] * 5
    + df["Projects"] * 2
)

# Add randomness
score += np.random.normal(0, 15, n)

# Use the median as the placement threshold
threshold = score.median()

df["Placement_Status"] = np.where(
    score >= threshold,
    "Placed",
    "Not Placed"
)

# Shuffle the dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save dataset
df.to_csv("data/student_data.csv", index=False)

print("Dataset created successfully!")
print(f"Total students: {len(df)}")

print("\nFirst 5 records:")
print(df.head())

print("\nPlacement distribution:")
print(df["Placement_Status"].value_counts())

print("\nDataset saved to:")
print("data/student_data.csv")