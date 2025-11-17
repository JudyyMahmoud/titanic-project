import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from pathlib import Path
from typing import Tuple


def load_data(filename: str = 'Titanic-Dataset.csv') -> pd.DataFrame:
    """Load the Titanic dataset from the repository's data folder or the repo root.

    Looks for these paths (in order):
    - <repo_root>/data/<filename>
    - <repo_root>/<filename>
    - <filename> (as provided)
    """
    repo_root = Path(__file__).resolve().parent.parent
    candidates = [repo_root / 'data' / filename,
                  repo_root / filename, Path(filename)]

    for p in candidates:
        if p.exists():
            df = pd.read_csv(p)
            print(f"Loaded dataset from: {p}")
            return df

    checked = '\n'.join(str(p) for p in candidates)
    raise FileNotFoundError(
        f"Could not find '{filename}'. Searched the following paths:\n{checked}\n"
    )


def preprocess_data(raw_df: pd.DataFrame, *, scale: bool = True) -> pd.DataFrame:
    """Preprocess the Titanic dataframe and return a cleaned, ML-ready DataFrame.

    - fills missing Age with mean
    - fills missing Embarked with mode
    - encodes Sex and Embarked into dummies
    - drops unused/text columns
    - scales numeric columns (optional)
    """
    df = raw_df.copy()

    # Basic checks
    if 'Age' in df.columns:
        df['Age'] = df['Age'].fillna(df['Age'].mean())

    if 'Embarked' in df.columns:
        most_common_port = df['Embarked'].mode()[0]
        df['Embarked'] = df['Embarked'].fillna(most_common_port)

    # Encoding
    if 'Sex' in df.columns:
        sex_dummies = pd.get_dummies(df['Sex'], prefix='Sex', drop_first=True)
        df = pd.concat([df, sex_dummies], axis=1)

    if 'Embarked' in df.columns:
        embarked_dummies = pd.get_dummies(
            df['Embarked'], prefix='Embarked', drop_first=True)
        df = pd.concat([df, embarked_dummies], axis=1)

    # Drop original text columns that are not useful for modeling
    drop_cols = [c for c in ['Sex', 'Embarked', 'Name',
                             'Ticket', 'Cabin', 'PassengerId'] if c in df.columns]
    df = df.drop(columns=drop_cols)

    # Scale numeric columns
    if scale:
        columns_to_scale = [c for c in ['Pclass', 'Age',
                                        'SibSp', 'Parch', 'Fare'] if c in df.columns]
        if columns_to_scale:
            scaler = StandardScaler()
            df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

    return df


def plot_eda(df: pd.DataFrame) -> None:
    """Optional: quick exploratory plots. Call explicitly from a notebook or script."""
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))

    if 'Age' in df.columns:
        sns.violinplot(data=df, x='Age', ax=axes[0, 0])
        axes[0, 0].set_title('Age Distribution')
    else:
        axes[0, 0].set_visible(False)

    if 'Sex' in df.columns and 'Survived' in df.columns:
        sns.countplot(data=df, x='Sex', hue='Survived', ax=axes[0, 1])
        axes[0, 1].set_title('Survival count by sex')
    else:
        axes[0, 1].set_visible(False)

    if 'Embarked' in df.columns and 'Survived' in df.columns:
        sns.barplot(x='Embarked', hue='Survived', data=df, ax=axes[1, 0])
        axes[1, 0].set_title('Survival count')
    else:
        axes[1, 0].set_visible(False)

    if 'Pclass' in df.columns and 'Survived' in df.columns:
        sns.barplot(x='Pclass', y='Survived', data=df, ax=axes[1, 1])
        axes[1, 1].set_title('survival rate by class')
    else:
        axes[1, 1].set_visible(False)

    fig.suptitle('Titanic analysis dashboard', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.show()


if __name__ == '__main__':
    # Convenience runner for local debugging: load, preprocess, and (optionally) plot
    df = load_data()
    processed = preprocess_data(df)
    print(processed.head())
