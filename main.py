import visualise
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor


DISPLAY = True
FILE_LOC = r"insurance.csv"


if __name__ == '__main__':

    if DISPLAY:
        visualise.run(FILE_LOC)

    # Read in data from csv
    df = pd.read_csv(FILE_LOC)

    # data processing
    df['smoker'] = df['smoker'].replace({'Yes': 1, 'No': 0})
    df['sex'] = df['sex'].replace({'male': 1, 'female': 0})

    # Feature engineering

    # Model Building
    X = df[['age', 'sex', 'bmi', 'children']]
    y = df['charges']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Outlier detection using Local Outlier Factor (LOF)
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
    outlier_labels = lof.fit_predict(X_train)

    # Filtering outliers from the training set
    X_train = X_train[outlier_labels != -1]
    y_train = y_train[outlier_labels != -1]

    # Initialize and train a model (example: Logistic Regression)
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate and print the mean squared error (MSE) on the test set
    mse = mean_squared_error(y_test, y_pred)

    print(mse)


