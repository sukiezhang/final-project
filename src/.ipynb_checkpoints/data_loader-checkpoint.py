# src/data_loader.py

import pandas as pd
import os

def list_files(directory):
    """List all files in the given directory."""
    for dirname, _, filenames in os.walk(directory):
        for filename in filenames:
            print(os.path.join(dirname, filename))

def load_data(train_path, test_path, submission_path):
    """Load train, test, and sample submission datasets."""
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    sample_submission = pd.read_csv(submission_path)
    return train, test, sample_submission
