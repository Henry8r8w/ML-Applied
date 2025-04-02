
## Identifying Missing Values
To deal with missing values in your dataset
- You can drop them
- You can do ordinal encoding (give rankign strength to each category)
- You can do onehot encoding (give 1 is the presense of the value, 0 to the basense of the value in matrix representation)
### Obtain Your Missing Values
```python
# Method 1: collect the  the pd.Series columns names if a null value is found within the column and drop that column
cols_with_missing = [col for col in X_train.columns
                     if X_train[col].isnull().any]
# Method 2: perform summazation accross the columns and obtain list form of pandas index (aka.column) if there is at least 1 null value
missing_val_count_by_column = df.isnull().sum() # give you pd.Series (col, null_sum)
cols_with_missing = missing_val_count_by_column[missing_val_count_by_column > 0].index.tolist()
```
Note:
- your X_train / X_valid +  y_train / y_valid should be given by your train-test split
    - X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                      random_state=0)
## Approach 1: Drop Columns with Missing Values
### Implementation
```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

def score_dataset(X_train, X_valid, y_train, y_valid, model=RandomForestRegressor(n_estimators=10, random_state=0)):
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)
cols_with_missing = [col for col in X_train.columns if X_train[col].isnull().any()] # obtain the missing columns

# Perform Column Drops (axis = 1)
reduced_X_train = X_train.drop(cols_with_missing, axis=1)
reduced_X_valid = X_valid.drop(cols_with_missing, axis=1)

print("MAE using column drop:", score_dataset(reduced_X_train, reduced_X_valid, y_train, y_valid))
```

### Effectiveness and Considerations
- Simple to implement, no risk of introducing bias
- Loss of potentially valuable information, especially if columns have few missing values
- Generally the least effective approach (third best)

## Approach 2: Imputation - the art of adding values
### Implementation
```python
from sklearn.impute import SimpleImputer


my_imputer = SimpleImputer(strategy='mean')  # default is mean; can also use 'median', 'most_frequent', or 'constant'


imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train)) # apply imputation on training data and learn the imputing strategy
imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid)) # apply the same imputation strategy learned from fit_transform to validation data

# Column names get remove during imputation, so put them back
imputed_X_train.columns = X_train.columns
imputed_X_valid.columns = X_valid.columns

print("MAE using imputation:",score_dataset(imputed_X_train, imputed_X_valid, y_train, y_valid))
```

### Effectiveness and Considerations
- Preserves all columns and their information
- Imputed values might not accurately represent what's missing
- Often the most effective approach (best)

## Approach 3: Imputation + Indicator Columns
### Implementation
```python
# Make copies to avoid changing original data
X_train_plus = X_train.copy()
X_valid_plus = X_valid.copy()

# Create bool-indicator columns for each column with missing value (checked by isnull())
for col in cols_with_missing:
    X_train_plus[col + '_was_missing'] = X_train_plus[col].isnull()
    X_valid_plus[col + '_was_missing'] = X_valid_plus[col].isnull()

my_imputer = SimpleImputer()
imputed_X_train_plus = pd.DataFrame(my_imputer.fit_transform(X_train_plus))
imputed_X_valid_plus = pd.DataFrame(my_imputer.transform(X_valid_plus))

# Column Name Restore
imputed_X_train_plus.columns = X_train_plus.columns
imputed_X_valid_plus.columns = X_valid_plus.columns

print("MAE using imputation + indicators:",score_dataset(imputed_X_train_plus, imputed_X_valid_plus, y_train, y_valid))
```

### Effectiveness and Considerations
- Preserves all information while adding context about missingness patterns
- Increases dimensionality of the dataset
- Usually the second-best approach

## Additional Data Cleaning Techniques
### Handling Categorical Variables
```python
# Select only numerical columns
X_numerical = df.select_dtypes(exclude=['object'])
```

### Custom Imputation Strategies
```python
# Custom imputation for different columns
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# For numerical columns: use mean
# For categorical columns: use most frequent value
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean'))
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent'))
])

# Get lists of numeric and categorical columns
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X.select_dtypes(include=['object', 'category']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])
# Pipline methods (from your estimator/decsion_function()) : fit(), transform(), predict(), score()
preprocessor.fit(X_train) 
processed_X_train = preprocessor.transform(X_train)
processed_X_valid = preprocessor.transform(X_valid)
```

## Cheatsheet
### Appeared Functions
- `df.drop(columns, axis=1)`: drop columns (axis=1) or rows (axis=0)
- `df.shape`: check dimensions of DataFrame
- `df.isnull().sum()`: count missing values by column
- `ColumnTransformer`: apply different transformations to different columns
- `df.select_dtypes()`: filter columns by data type

### Strategy
1. Look into your data missingness patterns through null-summation pd indexing or isnull() column name collection
2. Try the 3 approaches and compare their performance on validation data
