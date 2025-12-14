# Random Forest Tree (Supervised ML, classification, predict a target class, i.e. categorical variable)
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# Import the file (Heart Disease Dataset)
# Use all features to predict a target
filepath = r"C:\Users\ksenia\Desktop\PythonProject\Heart.csv"
raw_df = pd.read_csv(filepath)

# Categorical features
# sex, cp, fbs, restecg, exang, slope, ca, thal
# Numerical features
# age, trestbps, chol, thalach, oldpeak

print(raw_df.info())
print(raw_df.head(3))
print(raw_df.isnull().sum())
raw_df.dropna(subset=['target'], inplace=True)
print(raw_df.columns.tolist())
print(raw_df['target'].value_counts())

# Split the dataset (60% Training, 20% Validation, 20% Test)
from sklearn.model_selection import train_test_split
# First, split 60% train, 40% temp
train_df, temp_df = train_test_split(
    raw_df, test_size=0.4, stratify=raw_df['target'], random_state=42)

# Then split temp into 50% validation, 50% test -> each 20% of original
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, stratify=temp_df['target'], random_state=42)

# Check sizes
print(train_df.shape, val_df.shape, test_df.shape)

# Define features and target to train the model
input_cols = list(train_df.columns)[0:-1]
target_col = 'target'

train_inputs = train_df[input_cols].copy()
train_targets = train_df[target_col].copy()

# Categorical features
categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

# Numerical features
numerical_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

# # Select the features from the DataFrame
categorical_df = train_inputs[categorical_features]
# numerical_df = train_inputs[numerical_features]
#
# # Skip the step of SimpleImputer as we don't have missing values in features
# # Before Training the model, see the distribution of features correlations (feature vs target)
# # Ideally, see all correlations between features
#
# # Normality test (Shapiro-Wilk) for the continous features
# from scipy.stats import shapiro
# for col in numerical_df:
#     stat, p = shapiro(numerical_df[col])
#     print(f"{col} - Stat={stat:.3f}, p={p:.6f} -> {'Normal' if p>0.05 else 'Non-normal'}")
#
# # Check the distribution of the values
# for col in numerical_df:
#     plt.figure()  # creates a new figure for each column
#     plt.hist(numerical_df[col], bins=50, color='skyblue', edgecolor='black')
#     plt.title(f"{col} distribution")
#     plt.xlabel(col)
#     plt.ylabel("Frequency")
#     plt.show()
#
# # Make a scatter plot numerical feature vs target
# for col in numerical_df:
#     plt.figure(figsize=(6,4))
#     plt.scatter(numerical_df[col], train_targets, alpha=0.7)
#     plt.title(f"{col} vs {target_col}")  # use the variable, not the string 'target'
#     plt.xlabel(col)
#     plt.ylabel(target_col)
#     plt.show()
#
# Calculate correlation coefficient between numeric non-normally distributed continous features vs binary target
# from scipy.stats import pointbiserialr
# # Create an empty list to store results
# results = []
#
# for col in numerical_df:
#     # Calculate point-biserial correlation
#     corr, p = pointbiserialr(train_targets, numerical_df[col])
#     # Append to results list
#     results.append({'Feature': col, 'PointBiserial_r': corr, 'p_value': p})
#
# # Convert to DataFrame
# corr_table = pd.DataFrame(results)

# # Here, you have to perform the same steps but for categorical variables
# from scipy.stats import chi2_contingency
# results = []
#
# for col in ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']:
#     # Build contingency table
#     contingency_table = pd.crosstab(raw_df['target'], raw_df[col])
#
#     # Chi-squared test
#     chi2, p, dof, expected = chi2_contingency(contingency_table)
#
#     results.append({
#         'Feature': col,
#         'Chi2_stat': chi2,
#         'p_value': p
#     })
#
# # Convert to DataFrame
# corr_table_cat = pd.DataFrame(results)
# print(corr_table_cat)
#
# contingency_table_cp = pd.crosstab(raw_df['target'], raw_df['cp'])
# print(contingency_table_cp)

# Scale all numeric features into (0, 1) range
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
train_inputs[numerical_features] = scaler.fit_transform(train_inputs[numerical_features])

# One-hot encoding of categorical columns
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(drop='first', sparse_output=False, dtype=int)
encoded = encoder.fit_transform(train_inputs[categorical_features])
encoded_df = pd.DataFrame(
    encoded,
    index=train_inputs.index,
    columns=encoder.get_feature_names_out(categorical_features))

train_inputs = train_inputs.drop(columns=categorical_features)
train_inputs = pd.concat([train_inputs, encoded_df], axis=1)

# Training and Visualizing Decision Trees
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(train_inputs, train_targets)
train_predictions = model.predict(train_inputs)
pd.Series(train_predictions).value_counts()
train_probability = model.predict_proba(train_inputs)

# Model Evaluation
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
accuracy_score(train_targets, train_predictions) # The training set accuracy is close to 100%, but we can't rely on solely on training set accuracy,
# We must evaluate the model on th validation set too
# Make the validation set in the same way
val_inputs = val_df[input_cols].copy()
val_targets = val_df[target_col].copy()

val_inputs[numerical_features] = scaler.fit_transform(val_inputs[numerical_features])

encoder_val = OneHotEncoder(drop='first', sparse_output=False, dtype=int)
encoded_val = encoder_val.fit_transform(val_inputs[categorical_features])
encoded_val_df = pd.DataFrame(
    encoded_val,
    index=val_inputs.index,
    columns=encoder_val.get_feature_names_out(categorical_features))

val_inputs = val_inputs.drop(columns=categorical_features)
val_inputs = pd.concat([val_inputs, encoded_val_df], axis=1)

# Re-calculate the accuracy score
model.score(val_inputs, val_targets)
# Accuracy score has become 86%
val_targets.value_counts()/len(val_targets)
# The model predicts '1' at 51% of cases, so the model is over-fitted, I have to simply it

from sklearn.tree import plot_tree, export_text
plt.figure()
plot_tree(model, feature_names=train_inputs.columns, max_depth=1, filled=True)

# Display max depth
model.tree_.max_depth # Max depth is 13
# Display the tree
from sklearn.tree import export_text
tree_text = export_text(model, max_depth=3, feature_names=train_inputs.columns)
print(tree_text[:5000])

# Show the importance of each feature
importance_df_tree = pd.DataFrame({
       'feature': train_inputs.columns,
       'importance': model.feature_importances_
 }).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 5))
plt.bar(importance_df_tree['feature'], importance_df_tree['importance'])
plt.xticks(rotation=90)
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.title("Feature Importances (Decision Tree)")
plt.tight_layout()
plt.show()
# The most important features are thal_2 (thalium strass test reversible defect),
# thalach (max heart rate achieved), oldpeak (ST depression induced by exercise), age, and chol (serum cholesterol)
# You can modify the model and use only these parameters to improve the model if necessary

# Display number of tree leaves
model.get_n_leaves() # 52 leaves

# Make a plot with error (1 - accuracy score) vs max depth (from 1 to 13)
# Define a function to train and evaluate a decision tree with given max_depth
def evaluate_tree(max_depth, train_inputs, train_targets, val_inputs, val_targets):
    """
    Trains a Decision Tree with a given max_depth and returns:
    - training error rate
    - validation error rate
    """
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    model.fit(train_inputs, train_targets)
    # Predictions
    train_pred = model.predict(train_inputs)
    val_pred = model.predict(val_inputs)
    # Error rates
    error_train = 1 - accuracy_score(train_targets, train_pred)
    error_val = 1 - model.score(val_inputs, val_targets)
    return error_train, error_val

results = []
for depth in range(1, 14):
    error_train, error_val = evaluate_tree(depth, train_inputs, train_targets, val_inputs, val_targets)
    results.append({'max_depth': depth, 'train_error': error_train, 'val_error': error_val})

df_errors = pd.DataFrame(results)

# Plot both training and validation error
plt.figure(figsize=(8, 5))
plt.plot(df_errors['max_depth'], df_errors['train_error'], marker='o', label='Train Error')
plt.plot(df_errors['max_depth'], df_errors['val_error'], marker='o', label='Validation Error')
plt.xlabel('Max Depth')
plt.ylabel('Error Rate (1 - Accuracy)')
plt.title('Max Depth vs Error Rate')
plt.xticks(range(1, 14))
plt.legend()
plt.grid(True)
plt.show()

# Max depth returning the best accuracy in both training and validation sets is 3
# Remake the model with this max_depth = 3
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(train_inputs, train_targets)
train_predictions = model.predict(train_inputs)
print(accuracy_score(train_targets, train_predictions)) # the accuracy score is 83.25%
print(model.score(val_inputs, val_targets)) # the accuracy score is 82.93%

# The overfitting of the model is corrected
# Use GridSearchCV and RandomiserSearchCV to find the best composition of hyperparameters
# Define the parameter grid
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
    'min_samples_split': [2, 5, 10],
    'max_leaf_nodes': [None, 5, 10, 20, 30, 40, 50]}

# Create GridSearchCV
grid_search = GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1)   # use all CPU cores

# Fit on training data
grid_search.fit(train_inputs, train_targets)
# Best hyperparameters
print("Best parameters:", grid_search.best_params_)
# Best cross-validation accuracy
print("Best CV accuracy:", grid_search.best_score_)
# Best trained model
best_model = grid_search.best_estimator_
# The best parameters are : max_depth is 11, max_leaf_nodes is None (no limit), and min_samples_split is 2}

# Check the model using validation set
new_val_predictions = best_model.predict(val_inputs)
print(accuracy_score(new_val_predictions, val_targets))
# Accuracy score is 84.39%, which is slightly higher than without adjusting the hyperparameters

# You can also explore the following parameters to improve the model
# n_estimators
# max_features
# test_params(min_samples_split, min_samples_leaf)
# min_imputity_increase
# bootstrap(max_samples)
# class weigth

# Random Forests
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_jobs = -1, random_state = 42)
model.fit(train_inputs, train_targets)
train_predictions = model.predict(train_inputs)
print(accuracy_score(train_targets, train_predictions))
print(model.score(val_inputs, val_targets))
# The score is 93.17%, which is better than in Decision Tree (even without adjusting the hyperparameters)

# Create GridSearchCV for Random Forest
param_grid = {
    'n_estimators': [50, 100, 200],        # Number of trees in the forest
    'max_depth': [None, 5, 10, 15],        # Maximum depth of each tree; None = no limit
    'min_samples_split': [2, 5, 10],       # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4],         # Minimum number of samples required to be at a leaf node
    'max_features': ['sqrt', 'log2', None]} # Number of features to consider at each split
                                            # 'sqrt' = square root of total features, 'log2' = log2 of total features, None = all features

# Create GridSearchCV for Random Forest
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1)  # use all CPU cores

# Fit on training data
grid_search.fit(train_inputs, train_targets)
# Best hyperparameters
print("Best parameters:", grid_search.best_params_)
# Best parameters: {'max_depth': 10, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100}
# Best cross-validation accuracy
print("Best CV accuracy:", grid_search.best_score_)
# The score is 96.42%
# Best trained model
best_model = grid_search.best_estimator_

# Check the model using validation set
new_val_predictions = best_model.predict(val_inputs)
print(accuracy_score(new_val_predictions, val_targets))
# The score is 95.12%