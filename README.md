# credit-risk-classification
 credit-risk-classification

The purpose of this analysis is to use a dataset of historical lending activity to build a model that can identify the creditworthiness of borrowers. The dataset includes following information regarding 77,536 stored loans:
1- The loan size
2- Interest rate of the loan
3- The borrower income
4- The total debt to income ratio
5- Number of accounts
5- Derogatory marks 
6- Total debt of a borrower
7- Loan status, healthy or high risk

The dataset consists of 75,036 and 2,500 entries for healthy and risky loans, respectively. The main goal of this model is to predict the risk associated with lending money to future borrowers. 

To achieve the goal of this project, it's determined to use LogisticRegression and resampling methods. The LogisticRegression method is a statistical method suitable to predict a categorial outcome variable with two possible values, healthy and risky loans (in this case). Also, the resampling method was used as the dataset is imbalances (more than 90% of data is for healthy loans). 

In order to build the LogisticRegression model using original data, following steps were taken:
1- Create a dataframe using the available data
2- Separate the dataframe into 2 different ones, loan status (y) and other variables (X)
3- Split the dataframes into training and testing datasets 
4- Fit a logistic regression model by using the training data
5- Make a prediction using the testing data
6- Evaluate the model’s performance by doing the following:
6-1- Calculate the accuracy score of the model to evaluate the performance of a classification method.
6-2- Generate a confusion matrix to evaluate the performance of a classification model.
6-3- Create a classification report to provide a comprehensive view of the model's performance.

In order to build the LogisticRegression model using resampled data, following steps were taken:
1- Use the RandomOverSampler module from the imbalanced-learn library to resample the original data 
2- Fit the model using the resampled training data
3- Make a prediction using the testing data
4- Evaluate the model’s performance by doing the following:
4-1- Calculate the accuracy score of the model to evaluate the performance of a classification method.
4-2- Generate a confusion matrix to evaluate the performance of a classification model.
4-3- Create a classification report to provide a comprehensive view of the model's performance.


## Results

* Machine Learning Model 1, LogisticRegression model using original data:


                precision    recall  f1-score   support

  Healthy Loan       1.00      1.00      1.00     18759
High Risk Loan       0.87      0.89      0.88       625

      accuracy                           0.99     19384
     macro avg       0.94      0.94      0.94     19384
  weighted avg       0.99      0.99      0.99     19384


* Machine Learning Model 2, LogisticRegression model using resampled data:


                    precision    recall  f1-score   support

  Healthy Loan ROS       1.00      1.00      1.00     18759
High Risk Loan ROS       0.87      1.00      0.93       625

          accuracy                           1.00     19384
         macro avg       0.94      1.00      0.96     19384
      weighted avg       1.00      1.00      1.00     19384
## Summary

The precision outcome for both of the models is the same. However, the 2nd model performs best die to following reasons:
1- The recall is 1 for both healthy and risky loans. The recall is a metric used to evaluate the performance of a classification model when the identification of positive instances is important.
2- The accuracy of the model, a metric to evaluate the performance of the model, has been improved and became 1. It means the model is 100% accurate in predictions across all classes, helathy and risky loans. 


Using this model really depends on many factors such as how well the economy is. The 2nd model has a recall value of 1 meaning that all the positive instances were correctly identified. However, both of the models failed to predict 13% of risky loans. This can be the result of the low number of data entries for risky loans. 

Considering the above mentioned facts, it really depends on the risk acceptance of decision makers. Also, it's recommended to perform an analysis determining the potentail profit and loss that can be accrued using this model. 

If you do not recommend any of the models, please justify your reasoning.