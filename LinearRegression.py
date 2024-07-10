import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score



def preprocess_data(df):
    
    label_encoder = LabelEncoder()

    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = label_encoder.fit_transform(df[col])

    x = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values


    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    return X_train, X_test, y_train, y_test

def train_linear_regression(X_train, y_train):
    reg = LinearRegression()
    reg.fit(X_train, y_train)
    return reg

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    return y_pred, r2

def main():
    df =pd.read_csv(r"Data\CompaniesData.csv")

    X_train, X_test, y_train, y_test = preprocess_data(df)


    reg_model = train_linear_regression(X_train, y_train)

    y_pred, r2 = evaluate_model(reg_model, X_test, y_test)

    print("R-squared score:", r2)

if __name__ == "__main__":
    main()
