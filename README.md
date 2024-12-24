# IoT Intrusion Detection

This repository contains a Python script for detecting IoT intrusions using multiple machine learning classifiers. The script processes a dataset to classify network activities into normal or malicious categories based on feature inputs.

---

## Features

- **Machine Learning Models Implemented:**
  - Random Forest Classifier
  - Artificial Neural Network (ANN)
  - Naive Bayes Classifier
  - Support Vector Machine (SVM)
- **Data Preprocessing:**
  - Label Encoding
  - Feature Scaling (StandardScaler)
- **Metrics for Evaluation:**
  - Accuracy
  - Precision
  - Recall
  - F1 Score

---

## Prerequisites

To run this code, you need:

- Python 3.6+
- Required libraries:
  - `numpy`
  - `pandas`
  - `scikit-learn`

Install the required libraries using:
```bash
pip install numpy pandas scikit-learn
```

---

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/IoT-Hacking-Detection.git
   cd IoT-Hacking-Detection
   ```

2. Place your dataset in the repository and update the dataset path in the script:
   ```python
   dataset = pd.read_csv('<your-dataset-path>.csv')
   ```

3. Run the script:
   ```bash
   python main.py
   ```

4. The script will train and evaluate the following classifiers on your dataset:
   - Random Forest
   - Artificial Neural Network
   - Naive Bayes
   - Support Vector Machine

5. Results are displayed in the terminal, including metrics for both the training and test sets.

---

## File Structure

- `main.py`: Main Python script for intrusion detection.
- `<your-dataset>.csv`: Input dataset (replace with your file).

---

## Example Output

For each classifier, the script displays metrics such as:

```plaintext
Random Forest - Training Metrics:
Accuracy: 1.0000
Precision: 1.0000
Recall: 1.0000
F1 Score: 1.0000

Random Forest - Test Metrics:
Accuracy: 0.9900
Precision: 0.9870
Recall: 0.9900
F1 Score: 0.9880
```
