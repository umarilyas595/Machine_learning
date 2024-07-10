import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor


def preprocess_data(df):
    df.dropna(subset=['Star Rating', 'Rating', 'Staff', 'Facilities', 'Location', 'Comfort', 'Cleanliness'], inplace=True)
    df.drop(columns=['Hotel Names', 'Airport Shuttle'], inplace=True)
    label_encoder = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = label_encoder.fit_transform(df[col])

    x = df.drop(['Price Per Day ($)'], axis=1).values
    y = df['Price Per Day ($)'].values
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test 

def feature_correlation(df, target_col='Price Per Day ($)'):
    corr_matrix = df.corr()
    price_corr = corr_matrix[target_col].drop(target_col)
    print("Correlation of 'Price Per Day ($)' with other attributes:")
    print(price_corr.sort_values(ascending=True))

   

def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

def train_linear_regression(X_train, y_train, X_test):
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    return lr, y_pred

def evaluate_model(y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mae, mse, r2

def cross_validation(model, X_train_scaled, y_train):
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    return np.mean(cv_scores)

def train_random_forest(X_train_scaled, y_train, X_test_scaled):
    rf = RandomForestRegressor(n_estimators=30, random_state=42)
    rf.fit(X_train_scaled, y_train)
    y_pred = rf.predict(X_test_scaled)
    return rf, y_pred
def main():
    df = pd.read_csv(r".\Data\HotelsData.csv")

    X_train, X_test, y_train, y_test = preprocess_data(df)
    feature_correlation(df)

    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)

    lr_model, lr_y_pred = train_linear_regression(X_train_scaled, y_train, X_test_scaled)
    lr_mae, lr_mse, lr_r2 = evaluate_model(y_test, lr_y_pred)

    print("Linear Regression Model:")
    print(f"Mean Absolute Error (MAE): {lr_mae:.4f}")
    print(f"Mean Squared Error (MSE): {lr_mse:.4f}")
    print(f"R-squared (R2): {lr_r2:.4f}")
    lr_cv_r2 = cross_validation(lr_model, X_train_scaled, y_train)
    print(f"Cross-validation R-squared (Linear Regression): {lr_cv_r2:.4f}")


    rf_model, rf_y_pred = train_random_forest(X_train_scaled, y_train, X_test_scaled)
    rf_mae, rf_mse, rf_r2 = evaluate_model(y_test, rf_y_pred)


    print("\nRandom Forest Model:")
    print(f"Mean Absolute Error (MAE): {rf_mae:.4f}")
    print(f"Mean Squared Error (MSE): {rf_mse:.4f}")
    print(f"R-squared (R2): {rf_r2:.4f}")
    rf_cv_r2 = cross_validation(rf_model, X_train_scaled, y_train)
    print(f"Cross-validation R-squared (Random Forest): {rf_cv_r2:.4f}")

if __name__ == "__main__":
    main()
