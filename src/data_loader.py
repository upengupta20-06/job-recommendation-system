import pandas as pd

def load_data():
    jobs = pd.read_csv("data/jobs.csv")
    interactions = pd.read_csv("data/interactions.csv")
    return jobs, interactions
