# Insurance Statistical Analysis Project
## Overview
This repository contains the code and analysis for an insurance statistical analysis project aimed at understanding various factors influencing insurance premiums and predicting policy charges. The analysis involves exploring a diverse dataset from Kaggle, conducting statistical analysis, and developing predictive models.

## Data Source
The dataset used in this project is sourced from [Kaggle](https://www.kaggle.com/datasets/teertha/ushealthinsurancedataset/). 

## Data Insights
Size: The dataset consists of 1338 instances and 4 features.
Features: Includes information such as age, BMI, smoker status, region, number of children, and charges.
Target Variable: The primary focus is on 'charges,' representing insurance premiums or medical costs.


## Data Analysis
Outliers in Premium Charges: Notably, the dataset exhibits numerous outliers in charges, potentially influencing predictive modeling and statistical analysis.
Challenges and Considerations
High Variability: The presence of outliers and high variability in charges as well as BMI might impact the accuracy of statistical models and predictions.
Handling Outliers: Strategies for handling outliers, such as outlier detection and rejection.


## Data concerns
-It is not clear whether the charges in the data are costs to the insurance company or is the premium cost.
Given the very high variance of the data it appears to be medical payouts. 
-In addition, if the charges represent medical payouts, not premiums then there is an insufficient dataset to represent 
all outcomes and naturally the error of the model will be higher than otherwise.
-The charges involved can be quite large which overall pushs the MSE up quite high


## Results and Analysis
The end results were a mean square error of 130842965.

## Conclusion
The analysis highlights the complexity of insurance pricing, with substantial variability and outliers in premium charges. Efforts were made to address these challenges during data preprocessing and modeling to enhance the robustness and accuracy of predictive models.

## Repository Structure
data/: Contains the dataset files used in the analysis.
notebooks/: Jupyter notebooks detailing data exploration, statistical analysis, and model development.
src/: Source code files for data preprocessing, modeling, and analysis.
results/: Includes result summaries, visualizations, and model evaluation metrics.