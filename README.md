# US-Accidents

Context & Objectives:

The number of accidents is increasing taking away lives more every year. The objectives are to explore the underlying reasons behind accidents and predict both the severity and the accidents' duration to better deploy the resources available 

I. Data Preprocessing and Visualization

The initial dataset contained around 3 million observations. A sample of 500000 was taken and processed. The missing values were replaced by the median and then the finalized dataset was used for the analysis

The visual insights can be found in the notebook for a better understanding of the dataset.

As part of data exploration, text mining was used to explore the description dataset and understand which words have a higher frequency and come up more in accidents

II. Causal Inference

Causal inference is used to inspect causality between some factors such as weather and location in relation with the accident's severity. The treatments tested are: 'Exit', 'Visibility', 'Traffic_Signal' , 'Crossing', 'Precipitations' and 'Wind Speed'. The method used is linear regression and 2 features are obtained to be significant (p-value < 0.001) which are the presence of a traffic signal and the crossing with estimates of -0.269 and -0.246 respectively.

III. Feature Selection

Due to the large number of features in the dataset, three different methods are used to asses the importance of the features in relation with the severity of the accidents: RFE, LASSO and Random Forest. From the ranking obtained, we can identify the predictors related to both weather and traffic condition that may have a higher impact on severity. After obtaining a ranking of the most important features, random forest features with the highest gini are selected and used in the classification task:

IV. Classification & Regression

Developing a model that classifies the severity of accidents and predicts the duration the accidents is going to takes can be critical in order to help the concerned authorities better manage traffic when an accident occur and anticipate the needed resources to deal efficiently with accidents

    1. Classification
      
      Four machine learning algorithms are trained, and the accuracy is used to measure the performance
        Logistic Regression: 66.99%
        KNN: 66.04%
        Random Forest: 69.03%
        SVM: 66.24%
        
     Random Forest is the best performing algorithm with an accuracy of 69.03%
     
     2. Regression
     
       For the regression task, H2O AutoML is used to train and find the best possible model based on the MSE.
       This code can be found in a different notebook called AutoML
       The best model is called Stacked Ensemble with an RMSE of 3775
       

 
 

  
 




