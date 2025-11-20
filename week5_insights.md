# Week 5 — Model Optimization & Feature Insights

## 1. Introduction
This week’s challenge focused on optimizing machine learning models using:
- Logistic Regression  
- Random Forest  
- XGBoost  

The workflow involved hyperparameter tuning with GridSearchCV, performance comparison using cross-validation, and model interpretability through feature importance and SHAP visualizations.

---

## 2. Model Training & Tuning

### Hyperparameter Optimization
All models were tuned using **GridSearchCV** with **5-fold cross-validation**.  
Key parameters explored:

- **Logistic Regression:**  
  `C`, `penalty`

- **Random Forest:**  
  `n_estimators`, `max_depth`, `min_samples_split`

- **XGBClassifier:**  
  `learning_rate`, `max_depth`, `n_estimators`

This ensured each model was evaluated under multiple configurations to find the best combination of hyperparameters.

---

## 3. Performance Comparison

| Model | CV Mean Accuracy | Test Accuracy | Best Params |
|-------|------------------|---------------|-------------|
| **Logistic Regression** | 0.980220 | 0.973684 | `{'model__C': 0.1, 'model__penalty': 'l2'}` |
| Random Forest | 0.951648 | 0.947368 | `{'model__max_depth': 5, 'model__min_samples_split': 2, 'model__n_estimators': 100}` |
| XGBoost | 0.969231 | 0.947368 | `{'model__learning_rate': 0.1, 'model__max_depth': 3, 'model__n_estimators': 100}` |

**Best Model:** Logistic Regression  
It achieved the highest cross-validation accuracy and highest test accuracy among the models evaluated.

---

## 4. Interpretation

### 4.1 Feature Importance (Logistic Regression)
Logistic Regression coefficients revealed key influences:

- **Strong positive coefficients:**  
  `mean compactness`, `compactness error`, `fractal dimension error`  
  → Higher values increase likelihood of the **positive class (benign)**.

- **Strong negative coefficients:**  
  `mean radius`, `mean perimeter`, `worst radius`, `worst perimeter`  
  → Higher values push predictions towards the **negative class (malignant)**.

These relationships align with medical intuition: larger radius/perimeter values typically indicate malignancy.

---

### 4.2 SHAP Values
The SHAP summary plot confirmed:

- **Most influential features** matched the largest absolute logistic regression coefficients.
- **Directionality** was consistent:
  - Features with positive coefficients push predictions toward *benign*.
  - Negative features push toward *malignant*.
- **Global explanation:** The model consistently relied on compactness-related features and radius/perimeter statistics when making predictions.

SHAP provided a deeper, instance-level explanation of **how** features influenced each prediction.

---

## 5. Reusable Pipeline
A reusable `train_model()` function was created to:

- Apply preprocessing  
- Run GridSearchCV  
- Perform cross-validation  
- Return the best model  
- Output performance metrics  

This makes the entire ML workflow repeatable, clean, and scalable for future datasets and tasks.

---

## 6. Conclusion
This project demonstrated:

- Effective use of GridSearchCV for model tuning  
- Side-by-side comparison of classical (LR) vs ensemble (RF, XGB) models  
- The value of both performance metrics and interpretability tools  
- How a reusable ML pipeline boosts productivity and consistency  

The optimized Logistic Regression model delivered the strongest performance on the Breast Cancer dataset with interpretable outputs.

---

# What I Learned This Week

### 1. Hyperparameter tuning in practice
I learned how hyperparameters directly affect model performance and how GridSearchCV systematically tests combinations instead of guessing.

### 2. The importance of cross-validation
I now understand why a model must be evaluated across multiple splits to avoid lucky/biased scores.

### 3. Real-world model comparison
I gained hands-on experience comparing classical algorithms (Logistic Regression) with more complex ensemble models (Random Forest, XGBoost).

### 4. Practical interpretability
Using:
- Logistic Regression coefficients  
- Feature importance  
- SHAP plots  

helped me understand **which features matter, and how they influence predictions**.

### 5. Reusable ML design
Building a reusable pipeline function showed me how professional ML workflows are structured: modular, scalable, and clean.

---

# What This Project Taught Me

- **High accuracy alone is not enough** — model interpretability matters, especially in sensitive domains like medical prediction.
- **Simpler models can outperform complex ones** when properly tuned.
- **Automated optimization tools (GridSearchCV + Pipeline)** are essential for reliable and repeatable training.
- **SHAP provides deeper explanations** than traditional feature importance.
- **Modular code saves time**, especially when experimenting with many models and datasets.

Overall, this project improved both my technical ML skills and my confidence in evaluating and explaining machine learning models.

