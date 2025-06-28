# Network Intrusion Detection Using Machine Learning

## Overview
This project aims to build an effective **Intrusion Detection System (IDS)** to detect and classify malicious network activities using **supervised machine learning** and **ensemble techniques**. The system is evaluated on the **CIC-IDS-2017** dataset, which includes diverse and realistic network attacks.

## Models Used
### Traditional Machine Learning Models:
- Decision Tree (DT)
- Random Forest (RF)
- K-Nearest Neighbors (KNN)
- Logistic Regression (LR)
- CatBoost Classifier

### 🤖 Deep Learning Models:
- Feedforward Neural Network (FNN)
- Long Short-Term Memory (LSTM)

### 🧩 Ensemble Model:
- Stacked model combining KNN, DT, LR, and CatBoost, with Random Forest as the meta-learner.

## 📊 Dataset
**CIC-IDS-2017**  
A comprehensive network traffic dataset from the Canadian Institute for Cybersecurity containing both normal and various attack types (DDoS, PortScan, BruteForce, etc.).

## ⚙️ Methodology
- **Data Preprocessing**: Cleaning, feature selection (30–40 key features), scaling, one-hot encoding for categorical data.
- **Class Imbalance Handling**: SMOTE, under-sampling, and class-weight adjustments.
- **Model Training & Tuning**: Hyperparameter optimization using `GridSearchCV` and `RandomizedSearchCV`.
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC, PR-AUC, Confusion Matrix.

## Results
| Model              | Accuracy | Precision | Recall | F1-Score |
|-------------------|----------|-----------|--------|----------|
| Decision Tree      | 90.4%    | 92%       | 90%    | 90%      |
| Random Forest      | 99.5%    | 100%      | 100%   | 100%     |
| KNN                | 97.3%    | 97%       | 97%    | 97%      |
| Logistic Regression| 44.7%    | 38%       | 45%    | 38%      |
| CatBoost           | 95.8%    | 95%       | 96%    | 95%      |
| **Stacked Ensemble** | **98.9%** | **98.8%**   | **98.9%** | **98.8%** |

## Key Takeaways
- **Ensemble models** outperformed both individual ML and deep learning models in detecting and classifying intrusions.
- **Tree-based models** (RF, CatBoost) were particularly effective due to their robustness and ability to handle diverse data types.
- **Deep learning models** underperformed due to overfitting and lack of temporal features in flow-based data.

## Future Scope
- Explore advanced ensemble strategies (e.g., weighted stacking).
- Implement **real-time IDS deployment**.
- Apply **Explainable AI (XAI)** for transparency.
- Use **raw packet data** for better DL performance.
- Design **hybrid ML-DL systems**.
- Introduce **adversarial training** and **online learning** for adaptability.

## References
- CIC-IDS-2017 Dataset: [https://www.unb.ca/cic/datasets/ids-2017.html](https://www.unb.ca/cic/datasets/ids-2017.html)
- CatBoost: [https://catboost.ai](https://catboost.ai)
- Alom et al., "Intrusion detection using deep learning with LSTM", Journal of Big Data, 2021.
- Breiman, “Random Forests”, Machine Learning, 2001.





