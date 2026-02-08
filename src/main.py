from data_loader import load_data
from content_based import ContentBasedRecommender
from collaborative import CollaborativeRecommender
from hybrid import HybridRecommender

def explain_match(user_skills, job_row):
    skills = user_skills.lower().split()
    job_text = (job_row["skills"] + " " + job_row["description"]).lower()

    matched = [skill for skill in skills if skill in job_text]

    if matched:
        return "Your skills match: " + ", ".join(matched)
    else:
        return "This job is related to your overall skill set."

def main():
    jobs, interactions = load_data()

    content_model = ContentBasedRecommender(jobs)
    collab_model = CollaborativeRecommender(interactions, jobs)

    hybrid_model = HybridRecommender(
        content_model, collab_model, jobs
    )

    # USER INPUT
    user_skills = input("Enter your skills: ")

    recommendations = hybrid_model.recommend(
        user_skills, top_n=5
    )

    print("\nTop Job Recommendations:\n")

    for _, row in recommendations.iterrows():
        print(f"Job Title: {row['job_title']}")
        print("What you will do:")
        print(row["description"][:200] + "...")

        explanation = explain_match(user_skills, row)
        print("Why this matches you:")
        print(explanation)

        print("-" * 50)

if __name__ == "__main__":
    main()
