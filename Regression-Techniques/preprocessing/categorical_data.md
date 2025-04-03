
## Approaches to Categorical Variables
To deal with categorical data in your dataset
- Drop Categorical Columns 
- Ordinal Encoding - (a numerical ranks to categories)
- One-Hot Encoding - (binary columns assign 1 only to the correpsonding label, else 0)

## Identifying Categorical Variables
### Finding Categorical Columns
```python
def score_dataset(X_train, X_valid, y_train, y_valid, model=RandomForestRegressor(n_estimators=10, random_state=0)):
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)

# Method 1: identify data type and store corr. columns into an iterable
object_cols = [col for col in X_train.columns if X_train[col].dtype == "object"] # obect here is considering string type

# Method 2: Obtain pd Series with object data type bool value to perform indexing
s = (X_train.dtypes == 'object')
object_cols = list(s[s].index)

# Find low cardinality  categorical columns
low_cardinality_cols = [cname for cname in X_train.columns 
                       if X_train[cname].nunique() < 10 and 
                       X_train[cname].dtype == "object"]

# Count unique values in each categorical column
object_nunique = {col: X_train[col].nunique() for col in object_cols}
print(dict(zip(object_cols, object_nunique))) # dict(zip()) is an intrestin pair that puts the zip() pair into dictionary object

# Count categorical columns with high cardinality
high_cardinality_count = sum(1 for v in object_nunique.values() if v > 10)
```
Note:
- Cardinality is defined as the variation within a single categorical label. Cardinality less than 10 are typically suitable enough for trianing use
## Approach 1: Drop Categorical Variables
### Implementation
```python
# Simply remove all categorical columns
drop_X_train = X_train.select_dtypes(exclude=['object'])
drop_X_valid = X_valid.select_dtypes(exclude=['object'])
```

### Effectiveness and Considerations
- Simplest approach but generally performs worst
- Loses potentially valuable information
- May be appropriate when categorical variables have little predictive power

## Approach 2: Ordinal Encoding
### Implementation
```python
from sklearn.preprocessing import OrdinalEncoder

# Make copy to avoid changing original data 
label_X_train = X_train.copy()
label_X_valid = X_valid.copy()


ordinal_encoder = OrdinalEncoder()
label_X_train[object_cols] = ordinal_encoder.fit_transform(X_train[object_cols]) # apply ordinary_encoding on training data and learn the imputing strategy
label_X_valid[object_cols] = ordinal_encoder.transform(X_valid[object_cols]) # apply the same ordinary_encoding strategy learned from fit_transform to validation data
```

### Handling Unseen Categories
```python
# Identify columns that can be safely ordinal encoded
good_label_cols = [col for col in object_cols if 
                  set(X_valid[col]).issubset(set(X_train[col]))]
        
# Problematic columns that will be dropped from the dataset
bad_label_cols = list(set(object_cols)-set(good_label_cols))
```

### Effectiveness and Considerations
- Works well for ordinal categories (categories with a meaningful order)
- May introduce false relationships for nominal categories
- Second-best approach generally
- Modifications are made in-place

## Approach 3: One-Hot Encoding
### Implementation
```python
from sklearn.preprocessing import OneHotEncoder


OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False) # unknown category becomes 0, sparse matrix will not produced (lists will be produced)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[object_cols]))
OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[object_cols]))

# Add back the index after transformation
OH_cols_train.index = X_train.index
OH_cols_valid.index = X_valid.index

# Remove categorical columns (will replace with one-hot encoded columns)
num_X_train = X_train.drop(object_cols, axis=1)
num_X_valid = X_valid.drop(object_cols, axis=1)

# Concatenate the one-hot encoded columns to numerical features
OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)

# Ensure all column labels are strings - the onehot encoder from sklearn like return numerical id 
OH_X_train.columns = OH_X_train.columns.astype(str)
OH_X_valid.columns = OH_X_valid.columns.astype(str)

```
Notes:
- `handle_unknown='ignore'`: Prevents errors when validation data contains categories not in training data
- `sparse=False`: Returns a numpy array instead of a sparse matrix

### Effectiveness and Considerations
- Generally the best approach for categorical variables
- Works well for nominal (unordered) categories
- Increases dimensionality (adds columns)
- Number of new columns = (number of rows × number of unique categories) - original categorical columns

## CheatSheet
### List Comprehensions
- format: `[expression for item in iterable if condition]`
- example: [col for col in df.columns if df[col].dtype == "object"]

### Pandas Operations
- `.drop(columns, axis=1)`: remove columns (axis=1 for columns, axis=0 for rows)
- `.drop(..., inplace=True)`: modify the DataFrame in place instead of creating a copy
- `.dtypes`: access data types of columns
- `.nunique()`: count unique values in a column
- `.select_dtypes(include=['int64', 'float64'])`: select columns of specific types
- `.select_dtypes(exclude=['object'])`: exclude columns of specific types

### Some Other Useful Functions
- `set(list1).issubset(set(list2))`: heck if all elements in list1 are in list2
- `dict(zip(keys, values))`: create a dictionary from two lists
- `pd.concat([df1, df2], axis=1)`: combine DataFrames horizontally
