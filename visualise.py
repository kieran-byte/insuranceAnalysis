import pandas as pd
import matplotlib.pyplot as plt


def run(fileLoc):
    df = pd.read_csv(fileLoc)

    print(df.head())

    for column in df.columns:
        if df[column].dtype in ['int64', 'float64']:  # Check for numerical columns
            plt.figure(figsize=(8, 6))
            plt.boxplot(df[column].dropna(), vert=False)

            if column == 'region':
                plt.boxplot(df[column].dropna(), vert=False)

            plt.title(f'Box and whiskers plot for {column}')
            plt.xlabel(column)
            plt.show()

    print(df)

