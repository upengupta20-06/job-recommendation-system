import numpy as np

class HybridRecommender:
    def __init__(self, content_model, collab_model, jobs):
        self.content_model = content_model
        self.collab_model = collab_model
        self.jobs = jobs

    def recommend(self, user_skills, top_n=5):
        # Get content scores
        content_scores = self.content_model.get_scores(user_skills)

        # Get collaborative scores
        collab_scores = self.collab_model.get_scores()

        # Ensure same length
        if len(collab_scores) != len(content_scores):
            collab_scores = np.zeros(len(content_scores))

        # Hybrid score
        hybrid_scores = 0.8 * content_scores + 0.2 * collab_scores

        # Get top indices
        top_indices = hybrid_scores.argsort()[::-1][:top_n]

        # Return full job rows (IMPORTANT)
        return self.jobs.iloc[top_indices]
