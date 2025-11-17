from typing import Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def train_and_evaluate_model(df) -> Dict[str, Any]:
    """Train two models (LogisticRegression and RandomForest) and return results.

    Expects a DataFrame where the target column is named 'Survived'.
    The function will split the data, train models, and return a dict with
    metrics for each model.
    """
    if 'Survived' not in df.columns:
        raise ValueError("Input DataFrame must contain a 'Survived' column")

    X = df.drop('Survived', axis=1)
    y = df['Survived']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    lr_preds = lr.predict(X_test)
    lr_acc = accuracy_score(y_test, lr_preds)

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_preds)

    results = {
        'logistic': {
            'model': lr,
            'accuracy': lr_acc,
            'classification_report': classification_report(y_test, lr_preds, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, lr_preds).tolist(),
        },
        'random_forest': {
            'model': rf,
            'accuracy': rf_acc,
            'classification_report': classification_report(y_test, rf_preds, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, rf_preds).tolist(),
        },
        'model_name': 'random_forest' if rf_acc >= lr_acc else 'logistic',
        'accuracy': max(rf_acc, lr_acc),
    }

    return results


if __name__ == '__main__':
    print('This module provides train_and_evaluate_model(df)')
