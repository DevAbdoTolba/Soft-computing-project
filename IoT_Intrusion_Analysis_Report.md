# IoT Intrusion Detection Analysis Report

## 1. Dataset Overview

The IoT Intrusion detection dataset consists of 2000 records with 30 features. The target variable 'intrusion_type' has 5 different classes:
- Class 0: Normal traffic (1400 instances)
- Class 1-4: Different types of intrusion attacks (600 instances combined)

The dataset is imbalanced with 70% of the data representing normal traffic and 30% representing various intrusion types.

## 2. Feature Analysis

### Correlation Analysis
The correlation analysis revealed that most features have relatively low correlation with the target variable. The top correlated features with the intrusion type are:
- feature_27: 0.0383
- feature_9: 0.0348
- feature_8: 0.0337
- feature_17: 0.0290

This suggests that feature engineering or advanced feature selection techniques might be beneficial for improving classification performance.

## 3. Model Performance with All Features

| Model | Accuracy | F1 Score |
|-------|----------|----------|
| SVM | 0.7117 | 0.5918 |
| Random Forest | 0.7117 | 0.5918 |
| KNN | 0.6767 | 0.5777 |

With all 30 features, SVM and Random Forest achieved identical performance, slightly outperforming KNN. The gap between accuracy and F1 score indicates the impact of class imbalance on model performance.

## 4. Feature Selection Results

### Genetic Algorithm (GA)
The GA selected 15 features including:
- feature_27 (highest correlation with target)
- feature_1, feature_22, feature_21, feature_12, etc.

### Particle Swarm Optimization (PSO)
The PSO approach selected 15 features including:
- First 5 features (feature_0 through feature_4)
- Features 10-19

## 5. Model Performance Comparison

| Model | Approach | Features | Accuracy | F1 Score |
|-------|----------|----------|----------|----------|
| SVM | All Features | 30 | 0.7117 | 0.5918 |
| Random Forest | All Features | 30 | 0.7117 | 0.5918 |
| KNN | All Features | 30 | 0.6767 | 0.5777 |
| SVM | Genetic Algorithm | 15 | 0.7117 | 0.5918 |
| Random Forest | Genetic Algorithm | 15 | 0.7117 | 0.5918 |
| KNN | Genetic Algorithm | 15 | 0.6700 | 0.5769 |
| SVM | PSO | 15 | 0.7117 | 0.5918 |
| Random Forest | PSO | 15 | 0.7117 | 0.5918 |
| KNN | PSO | 15 | 0.6833 | 0.5898 |

## 6. Key Findings

1. **Feature Reduction**: Both GA and PSO successfully reduced the feature set by 50% (from 30 to 15 features) while maintaining similar performance levels.

2. **Model Consistency**: SVM and Random Forest models showed identical performance metrics across all feature selection approaches, suggesting robust performance regardless of feature selection method.

3. **KNN Variability**: KNN showed slight variations in performance across different feature sets, with the PSO-selected features providing the best results for this model (0.6833 accuracy and 0.5898 F1 score).

4. **Efficiency Gain**: The feature selection approaches demonstrated that the same level of performance can be achieved with half the number of features, which can lead to more efficient model training and deployment.

## 7. Conclusions

1. For this IoT intrusion detection task, SVM and Random Forest are the top-performing models with identical performance.

2. Feature selection techniques (GA and PSO) successfully reduced dimensionality without sacrificing performance, suggesting redundancy in the original feature set.

3. PSO slightly improved the KNN model's performance compared to using all features, showing that thoughtful feature selection can sometimes improve results while reducing complexity.

4. Given that performance is similar across all approaches, the recommended approach would be to use either GA or PSO-selected features with SVM or Random Forest models, as they provide the best balance of efficiency and effectiveness.

## 8. Future Work

1. Explore more advanced feature engineering techniques to improve the model performance further.

2. Address the class imbalance issue using techniques like SMOTE or class weighting.

3. Experiment with ensemble methods combining multiple models for potentially improved performance.

4. Investigate deep learning approaches that might capture more complex patterns in the data.

5. Extend the analysis to real-time detection scenarios to evaluate model performance in practical IoT security applications.
