from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ContentBasedRecommender:
    def __init__(self, jobs):
        self.jobs = jobs.copy()

        # Combine important job text fields
        self.jobs["combined"] = (
            self.jobs["job_title"] + " " +
            self.jobs["skills"] + " " +
            self.jobs["description"]
        )

        # TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.job_vectors = self.vectorizer.fit_transform(
            self.jobs["combined"]
        )

    def get_scores(self, user_skills):
        user_vec = self.vectorizer.transform([user_skills])
        similarity = cosine_similarity(user_vec, self.job_vectors)
        return similarity[0]
