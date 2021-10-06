### TeleCom Churn Prediction 
### Project Description
The data for the current project has been obtained from a competition that was provided by the French telecommunications company Orange. The problem deals with customer data, so the data was preliminarily anonymized: any personal information that allows identifying users was removed from the dataset, and the names and descriptions of the features intended for making forecasts were not provided.

In the project, we are going to deal with a small dataset. It consists of 50 thousand objects and includes 230 features:
  - 190 Numeric Features 
  - 40 Categorical Features

The task of predicting churn is one of the most important subtasks in the field of working with audiences and is relevant not only for telecommunications companies, but also for most organizations providing services in the B2C segment (approx. often in B2B too, but in this case we mean a company by a client). Such tasks often arise in practice for telecommunications operators, cable TV providers, insurance companies, banks, large and medium-sized Internet portals, etc.

### Project Goal
The goal of the project is to **learn how to find users prone to churn.** If you learn to find such users with sufficient accuracy in advance, you can effectively manage the churn:
  - Identify the reasons for the churn
  - Help users at risk to solve their problems and tasks
  - Run retention campaigns

In terms of Machine Learning, we have to build predictive models - models that predict the likelihood that a user will leave the service. In the classical setting, probabilistic binary classification models are built, where the target class is users leaving the service. **The probability that the user belongs to the target class is the target value** - the probability of churn. Accordingly, the greater this probability, the more chances that the user will refuse to use our service.

### Project Tasks 
The following tasks were formulated to achieve the goal:
  1) Exploratory Data Analysis: features, size,  identification of patterns and structure in the data
  2) Feature Selection: determine the relevant features to classify users who will "leave" or "stay"
  3) Data Preprocessing: determine the best way to process the features (e.g. imputers, categorical encoders) and dealing with class imbalance (e.g. Random Over/Undersampling, SMOTE, etc.)
  4) Model Selection: determine a model that would best cope with the stated objective of the project
  5) Model Optimization: determine optimal model hyperparameters, providing the best classification quality

### Project Metrics
There is no need to determine a metric for the competition because it is already given: `ROC-AUC`. However, we will also use auxiliary metrics such as `Precision` and `Recall` to track `False Positive` and `False Negative` as well as `F1-Score`

All the metrcis were computed using `StratifiedCrossValidation (n_folds = 5)`

### Project Pipeline
In order to conduct fast and robust (i.e. no data leakage) features processing the following `Pipeline` was defined:

![Untitled Diagram (7)](https://user-images.githubusercontent.com/56967765/136209077-1b45bc64-186e-4216-bb8a-9e360f2ef034.png)

The missing values for numeric and categorical features were filled with `SimpleImputer`:
  - Categorical Features: `SimpleImputer (strategy = 'constant', fill_value = 'unknown')`
  - Numeric Features: `SimpleImputer (strategy = 'mean')`

Features Encoding and Scaling:
  - Features Encoding: `TargetEncoder()`
  - Features Scaling: `StandardScaler()`

All preprocessing elements were cross-validated and the best were chosen (i.e. providing the best `ROC-AUC` score). The above `Pipeline` accepts the final sample (i.e. **after a feature selection stage**) 

### Feature Selection 
It should be said that there were a lot of missing values in the dataset. The first step was to identify features that entirely consisted of missing values, such features were excluded from consideration. Further, a bar chart was plotted to visualize the ratio of missing values for each feature.

![Screenshot 2021-10-06 063123](https://user-images.githubusercontent.com/56967765/136212270-6639ecd8-1e54-417e-922c-e89dcbc84e62.png)

It was decided that features with a NaN ratio of more than 90% are excluded, since we are unlikely to be able to recover these dependencies. This made it possible to exclude a fairly large number of numerical features and a few categorical ones. The remaining features fell into further analysis.

Next, Categorical and Numeric features were analyzed separately to determine the relationship with the target feature:
  - Categorical Features: `V-Krammer value`
  - Numeric Features: `Chi-square value`

The calculated values were used to determine the top features. Since there were few of them, it was decided to iteratively add these features to the model, build it on the current set of features and assess the quality. The following results were obtained.

### Numeric Features

![Screenshot 2021-10-06 063614](https://user-images.githubusercontent.com/56967765/136213111-93c95142-9bdf-4778-b108-26de28bcc4bf.png)

It can be observed that the first top 15 features by correlation with the target provide an increase in quality, then the quality stops growing. Given this fact, the first top 15 features were selected.

### Categorical Features

![Screenshot 2021-10-06 064028](https://user-images.githubusercontent.com/56967765/136213985-0e1cf53c-8a88-49ae-b198-3d10a732e403.png)

These features were selected similarly.

### Features with Low Dispertion
The BoxPlot of numerical features was analyzed where it was found that the features `Var143` `Var173` `Var35` `Var132` differ little from zero. The exclusion of these features **led to an increase in the target metric.**

![Screenshot 2021-10-06 064451](https://user-images.githubusercontent.com/56967765/136214838-1827d407-3d1e-4885-86d5-20adfcf0e0c6.png)

### Learning Curves
At the initial stage, it was necessary to find out **if there is enough data for the model** and how the model copes with the classification. For this purpose, learning curves were built. Without feature selection "as is", the model had a problem with **High Variance**. To further improve the quality of the model, it would be necessary:
  - Collect more data (which is impossible)
  - Simplify the model, conduct feature selection
  - Apply regularization

This is how the **learning curves looked like without feature selection:**

![Screenshot 2021-10-06 064708](https://user-images.githubusercontent.com/56967765/136215267-cb90bc82-2f8a-4072-9ec2-08cacb164428.png)

**After feature selection**, the learning curves improved slightly, indicating correct feature selection.

![image](https://user-images.githubusercontent.com/56967765/136215386-289d0d4d-c65d-4f7f-b3f4-856a867ea763.png)

It has resulted in `ROC-AUC: 0.734913`

### Position of Objects in Feature Space and Outliers
It was also interesting to know how our objects are located in the feature space:
  - Are there many outliers?
  - Are classes separable well?

This is how the objects looked before feature selection. For visualisation, `Principal Component Analysis` was applied 

![image](https://user-images.githubusercontent.com/56967765/136215901-d1175124-2650-41bb-9f78-5e468e84bc8e.png)

You can see that there are outliers and the classes are strongly mixed with each other. After feature selection, we see a different picture:

![Screenshot 2021-10-06 065200](https://user-images.githubusercontent.com/56967765/136216221-4286cf21-9ad9-46e7-8c73-8bf026e6028c.png)

It is noticeable that the outliers have not gone anywhere. Experiments were carried out to eliminate outliers. Removing such observations from training led to a sharp drop in quality; an experiment was also conducted to replace such observations with an average, for example. The quality decreased slightly, but not significantly `ROC-AUC: 0.733` and the objects themselves in space took the following position:

![image](https://user-images.githubusercontent.com/56967765/136216493-b1c33a76-0817-4841-9533-ee50b19d11e3.png)

### Dealing with Class Imbalance 
The dataset is unbalanced. Class Ratio: `{-1.0: 0.924746, 1.0: 0.075254}`

The following **Undersampling Methods** were used:
  - `Random Undersampling`
  - `One-Sided Selection`
  - `Under-Sampling with Cluster Centroids`
  - `Nearmiss 1 & 2 & 3`
 
The following **Oversampling Methods** were used:
  - `Random Oversampling`
  - `SMOTE (Synthetic Minority Oversampling Technique)` (likely to be bad)
  - `ADASYN (Adaptive Synthetic Sampling Approach)`
  - `Combination SMOTE + Tomek Links`

These methods could not radically improve the situation. Three candidates were selected as the main ones:
  - Assigning weights to observations (`F1-Score: 0.282003`, `ROC-AUC: 0.733730`)
  - Random Oversampling (`F1-Score: 0.647494`, `ROC-AUC: 0.774711`)
  - Random Undersampling (`F1-Score: 0.284221`, `ROC-AUC: 0.733171`)

`Random Oversampling` **was not used in the final solution**, as there was a risk of overfitting due to duplicate objects. Finally, the method : Setting weights for observations (more reliable) was applied.

### Feature Importances
In the final decision, it was interesting to look at the importance of the features.

![Screenshot 2021-10-06 065855](https://user-images.githubusercontent.com/56967765/136217543-4a6e5dd6-a5dd-43e0-a954-ebca9b42e98f.png)

**Almost all of the selected features** are important for the model. The top 3 most important features are categorical features.

### Model Hyperparameters Optimization
The final model is: `Gradient Boosting Classifier`

The selection of model hyperparameters was carried out using the **Bayesian approach**, using the module `HyperOpt`. The number of parameters for the model is large, so using the standard `GridSearchCV` would be very **time-consuming.** The hyperparameter tweak only slightly improved the quality. The quality of the final solution was `ROC-AUC: 0.7364 (CV)`

Switching to `CatBoost` and optimizing its hyperparameters allowed to take (currently <a href='https://www.kaggle.com/c/telecom-clients-prediction2/leaderboard'>7th place</a> with the score: `0.73128`) 
