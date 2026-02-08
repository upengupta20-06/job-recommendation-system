import numpy as np
import pandas as pd

class CollaborativeRecommender:
    def __init__(self, interactions, jobs):
        self.jobs = jobs

        # Create user-job interaction matrix
        user_job_matrix = interactions.pivot_table(
            index="user_id",
            columns="job_id",
            values="interaction",
            fill_value=0
        )

        # Get all job_ids from jobs.csv
        all_job_ids = jobs["job_id"].values

        # Reindex the matrix to include all jobs at once
        self.user_job_matrix = user_job_matrix.reindex(
            columns=all_job_ids,
            fill_value=0
        )

    def get_scores(self):
        # Popularity score per job
        job_popularity = np.sum(
            self.user_job_matrix.values,
            axis=0
        )
        return job_popularity
