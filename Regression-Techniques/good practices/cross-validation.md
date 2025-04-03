
## Cross-Validation:The Iterative Nature of ML
Machine learning is fundamentally an iterative process. We typically:
1. Build models
2. Evaluate their performance 
3. Make adjustments
4. Repeat until satisfactory results are achieved

## Understanding Cross-Validation (CV)
Cross-validation divides your dataset into segments, using some segments for training and others for validation. This process is repeated multiple times with different segments serving as the validation set.
### How It Works
- Dataset is divided into `n` equal segments (folds)
- For each iteration:
  - 1/n of the data serves as validation set
  - (n-1)/n of the data serves as training set
- Process repeats n times with different validation segments
- Results are averaged across all iterations

### Benefits
- More stable and reliable performance estimates
- Uses all data for both training and validation
- Ideally, larger datasets produce more reliable CV results

## Implementing Cross-Validation with scikit-learn
```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score

my_pipeline = Pipeline(steps=[
    ('preprocessor', SimpleImputer()),
    ('model', RandomForestRegressor(n_estimators=50, random_state=0))
])

# Perform 5-fold cross-validation
# Note: scikit-learn uses negative MAE, so we multiply by -1 to get standard MAE
scores = -1 * cross_val_score(my_pipeline, X, y,
                             cv=5,  # 5-fold cross-validation
                             scoring='neg_mean_absolute_error')

print("MAE scores:\n", scores)

# Display Mean MAE across the CV
print("Average MAE score (across experiments):")
print(scores.mean())
```

## Key Parameters in cross_val_score

- `estimator`: the model/pipeline to evaluate
- `X`: feature data
- `y`: target data
- `cv`: number of folds (default is 5)
- `scoring`: performance metric to use ('neg_mean_absolute_error', 'accuracy', etc ...)
- `n_jobs`: number of parallel jobs (set to -1 to use all processors)

## Important Considerations
### Computational Cost
- Higher number of folds means more computation time
- Common choices: 5-fold or 10-fold CV
- Leave-one-out CV (n folds for n samples) is computationally expensive
