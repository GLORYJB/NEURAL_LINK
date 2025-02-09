# models/model.py

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_model(data, labels):
    """
    Train a machine learning model to classify neural signals.

    Parameters:
        data (pd.DataFrame): Feature data.
        labels (pd.Series): Target labels.

    Returns:
        RandomForestClassifier: Trained model.
    """
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    return model