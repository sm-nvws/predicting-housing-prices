import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasRegressor
from tensorflow.keras.optimizers import Adam
import statsmodels

df = pd.read_csv("NY-Housing-Dataset.csv")



print(df.info())
print(df.describe())


numeric_df = df.select_dtypes(include=[np.number])



sns.pairplot(df)
plt.ion()
plt.show()


df.boxplot(figsize=(15, 6))
plt.title('Box Plot of Features')
plt.show()

X = df[['PROPERTYSQFT']]  
y = df['PRICE']  


poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

selector = SelectKBest(f_regression, k='all') 
X_selected = selector.fit_transform(X_poly, y)


X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

def build_model():
    model = Sequential()
    model.add(Dense(64, input_dim=X_train_scaled.shape[1], activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model


nn_model = KerasRegressor(build_fn=build_model, epochs=100, batch_size=32, verbose=0)


rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)


from sklearn.ensemble import VotingRegressor
ensemble_model = VotingRegressor(estimators=[
    ('nn', nn_model),
    ('rf', rf_model),
    ('gb', gb_model)
])


param_grid = {
    'rf__n_estimators': [50, 100],
    'gb__n_estimators': [50, 100],
    'nn__batch_size': [16, 32]
}
grid_search = GridSearchCV(ensemble_model, param_grid, cv=3, scoring='neg_mean_squared_error', verbose=1)
grid_search.fit(X_train_scaled, y_train)


print(f'Best parameters: {grid_search.best_params_}')


y_pred = grid_search.best_estimator_.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f'MSE: {mse:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}')


plt.figure(figsize=(10, 6))
sns.residplot(x=y_test, y=y_pred, lowess=True, color="g")
plt.title('Residual Plot')
plt.xlabel('Observed Values')
plt.ylabel('Residuals')
plt.show()


cv_scores = cross_val_score(grid_search.best_estimator_, X_train_scaled, y_train, cv=5, scoring='r2')
print(f'Cross-validated R2 scores: {cv_scores}')


plt.figure(figsize=(10, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", linewidths=0.5)
plt.title('Feature Correlation Matrix')
plt.show()

corr_matrix = numeric_df.corr()



def predict_feature_value(corr_matrix, feature_a, feature_b, value_of_a):
  
  
    if feature_a not in corr_matrix.columns or feature_b not in corr_matrix.columns:
        raise ValueError(f"Either {feature_a} or {feature_b} does not exist in the dataset.")
    
    corr_ab = corr_matrix.loc[feature_a, feature_b]
    
    if np.isnan(corr_ab):
        raise ValueError(f"No correlation found between {feature_a} and {feature_b}.")
    
    mean_a = df[feature_a].mean()
    mean_b = df[feature_b].mean()
    std_a = df[feature_a].std()
    std_b = df[feature_b].std()

    predicted_b = mean_b + corr_ab * ((value_of_a - mean_a) / std_a) * std_b

    return predicted_b


# test
feature_a = 'PROPERTYSQFT' 
feature_b = 'PRICE'         
value_of_a = 3000           

predicted_value_of_b = predict_feature_value(corr_matrix, feature_a, feature_b, value_of_a)
print(f"Predicted value of {feature_b} given {value_of_a} of {feature_a}: {predicted_value_of_b}")





