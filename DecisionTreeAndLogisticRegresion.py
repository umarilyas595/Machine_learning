import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder


def preprocess_data(df):
    label_encoder = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = label_encoder.fit_transform(df[col])
    
    df.drop(columns=['PassengerId','Age','Name','SibSp','Parch','Ticket','Embarked','Cabin'], inplace=True)
    X = df.drop(['Survived'], axis=1)
    y = df['Survived']  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def standard_scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

def train_decision_tree(X_train, y_train):
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

def train_logistic_regression(X_train, y_train):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

def main():
    df = pd.read_csv(r"Data\TitanicData.csv")
    X_train, X_test, y_train, y_test = preprocess_data(df)
    

    
    dt_model = train_decision_tree(X_train, y_train)
    dt_pred = dt_model.predict(X_test)
    dt_accuracy = accuracy_score(y_test, dt_pred)
    print(f"Decision Tree Accuracy: {dt_accuracy:.4f}")
    
    X_train_scaled, X_test_scaled = standard_scale_data(X_train, X_test)
    lr_model = train_logistic_regression(X_train_scaled, y_train)
    lr_pred = lr_model.predict(X_test_scaled)
    lr_accuracy = accuracy_score(y_test, lr_pred)
    print(f"Logistic Regression Accuracy: {lr_accuracy:.4f}")

if __name__ == "__main__":
    main()
