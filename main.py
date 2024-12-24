import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

# Load the dataset
dataset = pd.read_csv('/Users/ziyadkhalid/Desktop/iot-hacking-detection/IoT_Intrusion 2.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Encode the class labels
le = LabelEncoder()
y = le.fit_transform(y)

# Feature scaling
sc = StandardScaler()
X[:, 1:] = sc.fit_transform(X[:, 1:])

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Define and train classifiers
classifiers = {
    "Random Forest": RandomForestClassifier(),
    "Artificial Neural Network": MLPClassifier(max_iter=300),  # Increase iterations for ANN
    "Naive Bayes": GaussianNB(),
    "Support Vector Machine": SVC()
}

# Evaluate each classifier
for name, clf in classifiers.items():
    try:
        clf.fit(X_train, y_train)

        # Training set evaluation
        y_train_pred = clf.predict(X_train)
        print(f"\n{name} - Training Metrics:")
        print(f"Accuracy: {accuracy_score(y_train, y_train_pred):.4f}")
        print(f"Precision: {precision_score(y_train, y_train_pred, average='weighted', zero_division=0):.4f}")
        print(f"Recall: {recall_score(y_train, y_train_pred, average='weighted'):.4f}")
        print(f"F1 Score: {f1_score(y_train, y_train_pred, average='weighted'):.4f}")

        # Test set evaluation
        y_test_pred = clf.predict(X_test)
        print(f"\n{name} - Test Metrics:")
        print(f"Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
        print(f"Precision: {precision_score(y_test, y_test_pred, average='weighted', zero_division=0):.4f}")
        print(f"Recall: {recall_score(y_test, y_test_pred, average='weighted'):.4f}")
        print(f"F1 Score: {f1_score(y_test, y_test_pred, average='weighted'):.4f}")

    except Exception as e:
        print(f"\n{name} encountered an error: {e}")
