# src/main.py

import pandas as pd
from preprocess import preprocess_data
from train import train_and_evaluate_model

# Define the path to your data
# Using a relative path from the root of the project is a good practice
DATA_PATH = r"C:\Users\judym\OneDrive\Desktop\python\Titanic_project\data\Titanic-Dataset.csv"


def run_pipeline():
    """
    This is the main function that runs the entire ML pipeline.
    """
    print("Starting the ML pipeline...")

    # Step 1: Load the raw data
    print(f"Loading data from {DATA_PATH}...")
    raw_df = pd.read_csv(DATA_PATH)

    # Step 2: Preprocess the data
    # The preprocess_data function should be in your src/preprocess.py file
    print("Preprocessing data...")
    processed_df = preprocess_data(raw_df)

    # Step 3: Train the model and get the results
    # The train_and_evaluate_model function should be in your src/train.py file
    print("Training and evaluating model...")
    results = train_and_evaluate_model(processed_df)

    # Step 4: Print the final results
    print("\n--- Pipeline Finished ---")
    print(f"Model: {results['model_name']}")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print("-------------------------")


if __name__ == "__main__":
    run_pipeline()
