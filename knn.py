import pandas as pd
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

def preprocess_data(df):

    df.drop(columns=['id','Unnamed: 32'],inplace=True)
    
    x = df.drop(['diagnosis'],axis=1)
    y = df['diagnosis']
    X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=2)
    return X_train, X_test, y_train, y_test

def standard_scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

def train_knn(X_train, y_train):
    model = KNeighborsClassifier()
    model.fit(X_train, y_train)
    return model

def main():
    df = pd.read_csv(r"Data\BreastCancerData.csv")
    X_train, X_test, y_train, y_test = preprocess_data(df)

    X_train_Scaled, X_test_Scaled = standard_scale_data(X_train,X_test)
 
    
    k_model = train_knn(X_train_Scaled,y_train)

    y_pred = k_model.predict(X_test_Scaled)

    print(f"Accuracy of model is:",accuracy_score(y_test, y_pred))

if __name__ == "__main__":
    main()






