# Steel Plate Fault Analysis Report

## 1. Dataset Overview

The Steel Plate Faults dataset consists of 1941 records with 27 features. The dataset contains measurements of steel plates with various types of faults.

### Fault Types Distribution:
- Other_Faults: 673 instances (34.67%)
- Bumps: 402 instances (20.71%)
- K_Scatch: 391 instances (20.14%)
- Z_Scratch: 190 instances (9.79%)
- Pastry: 158 instances (8.14%)
- Stains: 72 instances (3.71%)
- Dirtiness: 55 instances (2.83%)

## 2. Model Performance

### Results with All Features

| Model | Accuracy | F1 Score | Training Time | Prediction Time |
|-------|----------|----------|---------------|----------------|
| SVM | 0.7410 | 0.7412 | 0.0429s | 0.0394s |
| RandomForest | 0.7890 | 0.7891 | 0.3572s | 0.0088s |
| KNN | 0.7479 | 0.7454 | 0.0013s | 0.2314s |
### Results with Genetic Algorithm

| Model | Accuracy | F1 Score | Training Time | Prediction Time |
|-------|----------|----------|---------------|----------------|
| SVM | 0.6638 | 0.6607 | 0.0326s | 0.0284s |
| RandomForest | 0.7702 | 0.7700 | 0.2661s | 0.0092s |
| KNN | 0.6878 | 0.6870 | 0.0022s | 0.0179s |
### Results with PSO

| Model | Accuracy | F1 Score | Training Time | Prediction Time |
|-------|----------|----------|---------------|----------------|
| SVM | 0.7015 | 0.6994 | 0.0254s | 0.0253s |
| RandomForest | 0.7530 | 0.7532 | 0.2308s | 0.0091s |
| KNN | 0.7084 | 0.7110 | 0.0019s | 0.0177s |

## 3. Feature Selection

- All Features: Used all 27 features
- Genetic Algorithm: Selected 13 features based on importance
- PSO: Selected 10 features with different selection criteria

### Top 10 Features by Importance

| Feature | Importance |
|---------|------------|
| LogOfAreas | 0.0684 |
| Length_of_Conveyer | 0.0614 |
| Pixels_Areas | 0.0563 |
| Log_X_Index | 0.0534 |
| Steel_Plate_Thickness | 0.0491 |
| Outside_X_Index | 0.0478 |
| Minimum_of_Luminosity | 0.0446 |
| Orientation_Index | 0.0424 |
| X_Minimum | 0.0423 |
| Sum_of_Luminosity | 0.0405 |

## 4. Conclusions

- The best performing model was **RandomForest** using the **All Features** approach, achieving an accuracy of 0.7890 and an F1 score of 0.7891.

- Feature selection techniques (simulated GA and PSO) demonstrated that comparable performance can be achieved with fewer features (13 and 10 respectively) compared to using all 27 features.

- On average, the **All Features** approach yielded the best overall performance with an average accuracy of 0.7593 and an average F1 score of 0.7585.

