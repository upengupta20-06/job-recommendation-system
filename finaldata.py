import pandas as pd
import numpy as np

# Load original dataset
df = pd.read_csv("data/job_title_des.csv")

# Drop unwanted column
df = df.drop(columns=["Unnamed: 0"])

# Rename columns
df.columns = ["job_title", "description"]
# Create job_id
df["job_id"] = range(1, len(df) + 1)

# Use job title as skills (simple approach)
# Extract keywords from description as skills
# Extract basic skill keywords
# Extract basic skill keywords
skill_keywords = [
    "python", "java", "sql", "machine learning",
    "deep learning", "tensorflow", "pytorch",
    "excel", "power bi", "tableau",
    "django", "flask", "react",
    "html", "css", "javascript",
    "data analysis", "nlp", "opencv"
]

def extract_skills(text):
    text = str(text).lower()   # convert NaN to string
    found = [skill for skill in skill_keywords if skill in text]
    return " ".join(found)

# Apply extraction
df["skills"] = df["description"].apply(extract_skills)

# Replace empty or NaN skills with job title
df["skills"] = df.apply(
    lambda row: row["skills"] if row["skills"].strip() != "" else row["job_title"],
    axis=1
)

# Final safety step: remove any remaining NaN
df["skills"] = df["skills"].fillna("unknown")


jobs = df[["job_id", "job_title", "skills", "description"]]

# Remove duplicate job titles
jobs = jobs.drop_duplicates(subset="job_title")
jobs = jobs.reset_index(drop=True)

# Reassign job_id after removing duplicates
jobs["job_id"] = range(1, len(jobs) + 1)


# Save cleaned jobs dataset
jobs.to_csv("data/jobs.csv", index=False)

print("jobs.csv created successfully")

# ----------------------------
# Create synthetic interactions
# ----------------------------

num_users = 20
interactions = []

for user in range(1, num_users + 1):
    applied_jobs = np.random.choice(jobs["job_id"], size=3, replace=False)
    for job in applied_jobs:
        interactions.append([user, job, 1])

interactions_df = pd.DataFrame(
    interactions,
    columns=["user_id", "job_id", "interaction"]
)

interactions_df.to_csv("data/interactions.csv", index=False)

print("interactions.csv created successfully")
