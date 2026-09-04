import sqlite3

connection = sqlite3.connect("student_placement.db")

cursor = connection.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    age INTEGER,
    gender TEXT,
    cgpa REAL,
    tenth_percentage REAL,
    twelfth_percentage REAL,
    aptitude_score REAL,
    coding_score REAL,
    communication_score REAL,
    technical_skills INTEGER,
    certifications INTEGER,
    internship_experience INTEGER,
    projects INTEGER,
    prediction TEXT,
    probability REAL
)
""")

connection.commit()
connection.close()

print("Database created successfully!")