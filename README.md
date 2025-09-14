# Parkinson’s Disease Voice Detection Model

This project implements a machine learning model to detect Parkinson’s Disease (PD) using vocal biomarkers. The model analyzes features such as jitter, shimmer, and harmonics-to-noise ratio to identify early signs of PD. Using a Support Vector Machine (SVM) with an RBF kernel, the final model achieves **95.2% balanced accuracy** and **100% precision**, making it a promising tool for non-invasive early screening.

---

## Features

- **Non-Invasive Detection**: Identifies signs of Parkinson’s Disease using voice data alone.  
- **High Accuracy**: Achieves 95.2% balanced accuracy and zero false positives on the test set.  
- **Advanced Preprocessing**: Applies standardization and stratified train/test splitting to ensure model reliability.  
- **Predictive System**: Includes a real-time prediction function for single-sample inference.  
- **Model Optimization**: Uses GridSearchCV for hyperparameter tuning of the SVM classifier.  

---

## Dataset

The dataset includes **195 voice recordings** from **31 individuals** (23 with PD and 8 healthy controls), with features extracted from voice recordings of sustained phonation (vowel sounds), using metrics such as fundamental frequency, jitter, shimmer, and harmonics‑to‑noise ratio.  
Target variable: `status`  
- 0 = Healthy  
- 1 = Parkinson’s Disease  

[Kaggle Dataset – Parkinson’s Disease Data Set](https://www.kaggle.com/datasets/vikasukani/parkinsons-disease-data-set)

---

## Methodology

### Data Preprocessing

- Applied **`StandardScaler`** to normalize features.  
- Performed **stratified train/test split** to maintain class distribution.  
- Extracted key features: fundamental frequency (Fo, Fhi, Flo), jitter, shimmer, HNR, and more.

### Class Imbalance Handling

- Tested **SMOTE** to oversample the minority class (healthy).  
- Found it **reduced performance** due to small dataset size and synthetic noise, and was not used in the final model.

### Model Selection

- Chose **Support Vector Machine (SVM)** for its ability to handle high-dimensional, nonlinear data.  
- Compared **linear SVM** vs. **RBF kernel**; RBF performed significantly better.

### Hyperparameter Tuning

- Used GridSearchCV with 5-fold StratifiedKFold to identify the optimal combination of SVM hyperparameters, including C (regularization strength) and gamma (kernel coefficient).

### Performance Evaluation
  - **Balanced Accuracy**: 95.2%  
  - **Precision**: 100%

The perfect precision ensures the model avoids false positives, which is crucial for reducing unnecessary concern in healthy individuals. High balanced accuracy confirms strong performance across both classes, despite the dataset's imbalance.

---

## Predictive System

A function is included for real-time prediction from new samples:

```python
def predictive_system(input_data):
    arr = np.asarray(input_data).reshape(1, -1)
    arr_scaled = scaler.transform(arr)
    pred = model.predict(arr_scaled)[0]
    return "Parkinson's" if pred == 1 else "Healthy"
```

---

## Requirements

To run the project, you will need:
- Python 3.8+
- Libraries:
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `imbalanced-learn`
  - `matplotlib` and `seaborn` (optional for visualizations)
