Smart Chiller Load Estimation

Problem Statement: Consider the case of a hotel. Our objective is to estimate the chiller load, which is influenced by several factors, including hotel occupancy, outside weather conditions, the day of the week (weekday vs. weekend), and nearby attractions. By taking these factors into account, we aim to accurately estimate the chiller load and adjust the equipment operations accordingly. This approach allows for efficient system performance, avoiding unnecessary energy consumption from running the equipment at full capacity when it's not required.

Solution: 
We have developed two ML model, both are regression models , where Random Forest Regressor (an ensemble method) is used to perform Random Forest Regression Algorithm, which lower down the variance values of each decision trees trained parallely.
