#Feature Engineering

## Standardize
data has abnormal data and noisy

## Normalized
#### Interval scaling (Special Case)
data is stable with no extremely max and min value

## Quantitative feature binarization

## One Hot

## Missing value fill

## Data Convert
#### Convert to Polynomial
#### Log

## Dimensionality reduction
#### Feature Selection
eg. standard deviation is 0 (remove)  
    relative with target (maintain)

#### Specific Method
###### Filter: filter by threshold
Correlation coefficient  
Chi-square test  
Mutual information  
Information gain  
###### Package: according to target function score
target (AUC/MSE)
Full search  
Heuristic search  
Random search: Genetic Algorithm, Simulated Annealing
###### Embeddings
Normalization: L1 (LASSO), L2 (RIDGE)  
Decision Tree: Entropy, Information gain  
Deep Learning

## Feature Engineering
polynomial feature  
ratio feature  
absolute value  
max, min, xor  
median, mean, mode, min, max, std, var, freq  
decision tree(tree, GBDT, random forest)  
A * B, A * B * C, A * A  
bucket  
combiine one-hot  


## Common Tools
sklearn.feature_selection.VarianceThreshold (using std)  
...... .SelectKBest (Correlation coefficient method, Chi-square test, Mutual information)
...... .RFE (Recursive Elimination of Features)  
...... .SelectFromModel (using LogisticRegression, GBDT)
... .decomposition.PCA (Linearly)
... .discriminant_analysis.LinearDiscriminantAnalysis (LDA: Linearly)
