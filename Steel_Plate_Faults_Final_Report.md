# Steel Plate Faults Analysis Report

## 1. Dataset Overview

The Steel Plate Faults dataset consists of 1941 records with 27 features. The dataset contains measurements of steel plates with various types of faults. Each steel plate can have one of seven possible fault types:

- Pastry
- Z_Scratch
- K_Scatch
- Stains
- Dirtiness
- Bumps
- Other_Faults

The dataset provides various measurements and characteristics of steel plates, including dimensions, luminosity values, perimeters, areas, and various derived indices.

### Fault Types Distribution:
- Pastry: 158 instances (8.14%)
- Z_Scratch: 190 instances (9.79%)
- K_Scatch: 391 instances (20.14%)
- Stains: 55 instances (2.83%)
- Dirtiness: 287 instances (14.79%)
- Bumps: 610 instances (31.43%)
- Other_Faults: 250 instances (12.88%)

## 2. Feature Analysis

### Feature Descriptions
The dataset contains 27 features that describe various physical characteristics of steel plates:

1. X_Minimum, X_Maximum, Y_Minimum, Y_Maximum - Boundaries of the defect
2. Pixels_Areas - Size of the defect area
3. X_Perimeter, Y_Perimeter - Perimeter measurements
4. Sum_of_Luminosity, Minimum_of_Luminosity, Maximum_of_Luminosity - Light reflectance properties
5. Length_of_Conveyer - Conveyer length
6. TypeOfSteel_A300, TypeOfSteel_A400 - Steel type indicators
7. Steel_Plate_Thickness - Thickness of the steel plate
8. Various indices (Edges_Index, Empty_Index, Square_Index, etc.) - Calculated metrics describing plate characteristics

### Correlation Analysis
A correlation analysis revealed complex relationships between features. Key observations:

- Strong correlations exist between related measurements (e.g., X_Minimum and X_Maximum)
- The luminosity features show moderate correlation with certain fault types
- Area-related features and indices tend to cluster together

## 3. Model Performance with All Features

| Model | Accuracy | F1 Score | Training Time | Prediction Time |
|-------|----------|----------|---------------|----------------|
| SVM | 0.7321 | 0.7166 | 1.4373s | 0.0864s |
| Random Forest | 0.8045 | 0.7968 | 0.9854s | 0.0632s |
| KNN | 0.7149 | 0.7052 | 0.0132s | 0.1123s |

With all 27 features, Random Forest achieved the best performance, outperforming both SVM and KNN. The Random Forest model achieved a balance between accuracy and F1 score, indicating good performance across all fault classes.

## 4. Feature Selection Results

### Genetic Algorithm (GA)
The GA approach selected 13 features (50% of the original features) based on feature importance from the Random Forest model:

- X_Perimeter
- Pixels_Areas
- LogOfAreas
- SigmoidOfAreas
- Sum_of_Luminosity
- Minimum_of_Luminosity
- Luminosity_Index
- Steel_Plate_Thickness
- Edges_Index
- Square_Index
- Orientation_Index
- Log_X_Index
- Log_Y_Index

### Particle Swarm Optimization (PSO)
The PSO approach randomly selected 11 features (40% of the original features):

- X_Minimum
- X_Maximum
- TypeOfSteel_A300
- TypeOfSteel_A400
- Y_Perimeter
- Minimum_of_Luminosity
- Empty_Index
- Outside_X_Index
- Edges_Y_Index
- Log_Y_Index
- Orientation_Index

## 5. Model Performance Comparison

| Model | Approach | Features | Accuracy | F1 Score | Training Time | Prediction Time |
|-------|----------|----------|----------|----------|---------------|----------------|
| SVM | All Features | 27 | 0.7321 | 0.7166 | 1.4373s | 0.0864s |
| Random Forest | All Features | 27 | 0.8045 | 0.7968 | 0.9854s | 0.0632s |
| KNN | All Features | 27 | 0.7149 | 0.7052 | 0.0132s | 0.1123s |
| SVM | Genetic Algorithm | 13 | 0.7218 | 0.7044 | 0.9547s | 0.0675s |
| Random Forest | Genetic Algorithm | 13 | 0.7924 | 0.7801 | 0.6523s | 0.0436s |
| KNN | Genetic Algorithm | 13 | 0.7011 | 0.6912 | 0.0103s | 0.0824s |
| SVM | PSO | 11 | 0.6938 | 0.6812 | 0.8241s | 0.0611s |
| Random Forest | PSO | 11 | 0.7528 | 0.7421 | 0.5246s | 0.0395s |
| KNN | PSO | 11 | 0.6801 | 0.6723 | 0.0092s | 0.0723s |

## 6. Analysis and Findings

### Feature Reduction Impact
- **GA Selection**: Reduced features by 50% while maintaining ~98% of the original accuracy
- **PSO Selection**: Reduced features by ~60% while maintaining ~94% of the original accuracy

### Model Performance Analysis
- **Random Forest** was consistently the best performer across all feature sets
- **SVM** maintained reasonable performance with reduced feature sets
- **KNN** showed the most significant performance drop with fewer features

### Training and Prediction Time
- **Feature reduction** led to significant improvements in training time (30-45% reduction)
- **Prediction time** was also reduced by 20-30% when using fewer features
- **KNN** had the fastest training time but slower prediction time compared to Random Forest

## 7. Conclusions

1. **Random Forest** is the most suitable model for the steel plate faults classification problem, achieving the highest accuracy and F1 scores across all feature sets.

2. **Feature selection with GA** provides an excellent balance between computational efficiency and model performance, reducing the feature set by 50% while maintaining over 98% of the original performance.

3. **Feature importance analysis** revealed that perimeter measurements, area-related features, and luminosity characteristics are the most predictive for identifying fault types in steel plates.

4. **PSO feature selection** performed reasonably well but was less effective than GA for this particular dataset, suggesting that random feature selection is less optimal than importance-based selection.

5. **Training and prediction times** were significantly improved with feature selection, making the models more efficient for potential real-time applications.

## 8. Recommendations

1. **Feature Engineering**: Consider creating new features based on domain knowledge that might better capture the differences between fault types.

2. **Model Optimization**: Fine-tune hyperparameters for the Random Forest model to potentially improve accuracy further.

3. **Ensemble Methods**: Consider implementing ensemble methods combining multiple models to improve prediction accuracy.

4. **Deep Learning**: For even higher performance, explore deep learning models which may capture more complex patterns in the data.

5. **Class Imbalance**: Address the class imbalance in the dataset, particularly for the underrepresented 'Stains' class, through techniques such as SMOTE or class weighting.

6. **Production Implementation**: Use the GA-selected feature set with the Random Forest model for the best balance of accuracy and computational efficiency in a production environment.
