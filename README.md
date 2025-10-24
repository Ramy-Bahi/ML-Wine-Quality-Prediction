# 🍷 Wine Quality Classification — Machine Learning Models Comparison

This project applies and compares multiple **Machine Learning classification algorithms** on the [Wine Quality Dataset (Cortez et al., 2009)](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009).  
It’s part of my continuous journey to master **Machine Learning**, covering techniques from **logistic regression** to **ensemble learning** before transitioning into **Deep Learning**.

---

## 📘 Project Overview

The goal is to predict the **quality of red wine** (rated 3–8) based on physicochemical tests.  
Since the dataset is imbalanced, a **“quality label”** was engineered to categorize samples into:
- **Low quality** (≤4)  
- **Medium quality** (5–6)  
- **High quality** (≥7)

Several models were trained, tuned, and compared to identify the best performer for this multi-class classification problem.

---

## ⚙️ Technologies Used

- **Python 3.x**
- **NumPy**, **Pandas** → data manipulation  
- **Matplotlib**, **Seaborn** → data visualization  
- **Scikit-Learn** → preprocessing, model training, and evaluation  
- **Joblib** → model export  

---

## 🧩 Data Preprocessing Pipeline

1. **Feature Engineering**
   - Created new ratios like `acid_ratio` and `sulfur_ratio`
   - Encoded the `quality_label` column using label encoding

2. **Handling Imbalance**
   - Used `class_weight='balanced'` for certain models to improve fairness

3. **Feature Scaling**
   - Standardized features using `StandardScaler` for better model performance

4. **Model Training**
   - Trained multiple classifiers:
     - Logistic Regression
     - Multinomial Logistic Regression
     - K-Nearest Neighbors (KNN)
     - Decision Tree
     - Random Forest
     - Support Vector Machine (SVM)
     - Gaussian Naive Bayes

---

## 📊 Model Results

| Model | Accuracy | F1 Score |
|--------|-----------|----------|
| **Random Forest** | **0.7531** | **0.7454** |
| Support Vector Machine | 0.7406 | 0.7315 |
| Logistic Regression | 0.7343 | 0.7276 |
| Multinomial Logistic Regression | 0.7343 | 0.7276 |
| K-Nearest Neighbors | 0.7281 | 0.7202 |
| Naive Bayes | 0.7093 | 0.7005 |
| Decision Tree | 0.6906 | 0.6925 |

---

## 🧠 Model Performance Discussion

Among all models tested, **Random Forest** clearly stood out as the best-performing classifier.  
After hyperparameter tuning using `GridSearchCV`, the best configuration was:

```python
{'class_weight': 'balanced', 'max_depth': 20, 'min_samples_leaf': 4, 
 'min_samples_split': 10, 'n_estimators': 300}
```

This tuning improved its accuracy to 0.7875 and F1 score to 0.7786, surpassing all other models, including the optimized SVM which achieved 0.7250 accuracy and 0.7236 F1 score.

The Random Forest’s ensemble approach handled the feature complexity and data imbalance effectively, whereas SVM, despite tuning with polynomial and RBF kernels, struggled to generalize on the minority classes.

---

## 🚀 Next Steps

- Add **XGBoost** and **LightGBM** models for further comparison  
- Explore **SMOTE oversampling** for better balance between classes  
- Conduct **feature importance analysis** to identify top predictors of quality  
- Visualize **decision boundaries** for SVM and KNN  

---

## 🧾 Repository Structure

Wine-Quality-Classification/
│
├── data/
│   └── winequality-red.csv
│
├── notebooks/
│   └── wine_classification.ipynb
│
├── models/
│   ├── best_random_forest.pkl
│   └── scaler.pkl
│
├── requirements.txt
├── README.md
└── .gitignore

---

## 🧩 How to Run

```bash
git clone https://github.com/<yourusername>/Wine-Quality-Classification.git
cd Wine-Quality-Classification
pip install -r requirements.txt
jupyter notebook notebooks/wine_classification.ipynb
```

---

## 📊 Dataset

Available on Kaggle: [**Red Wine Quality (Cortez et al., 2009)**](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009)

---

## 👨‍💻 Author

**Rami Bahi**

🎓 Master’s Student in Artificial Intelligence

💻 Passionate about Machine Learning, Deep Learning & Web Development

---

## ⭐ If you like this project...

Give it a star ⭐ on GitHub to support my work and journey!
