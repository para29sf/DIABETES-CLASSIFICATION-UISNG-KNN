<div align="center">
  
[1]: https://github.com/Pradnya1208
[2]: https://www.linkedin.com/in/pradnya-patil-b049161ba/
[3]: https://public.tableau.com/app/profile/pradnya.patil3254#!/
[4]: https://twitter.com/Pradnya1208


[![github](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/c292abd3f9cc647a7edc0061193f1523e9c05e1f/icons/git.svg)][1]
[![linkedin](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/9f5c4a255972275ced549ea6e34ef35019166944/icons/iconmonstr-linkedin-5.svg)][2]
[![tableau](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/e257c5d6cf02f13072429935b0828525c601414f/icons/icons8-tableau-software%20(1).svg)][3]
[![twitter](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/c9f9c5dc4e24eff0143b3056708d24650cbccdde/icons/iconmonstr-twitter-5.svg)][4]

</div>

# <div align="center">Diabetes classification using KNN</div>

<div align="center">
<img src  ="https://github.com/Pradnya1208/Diabetes-classification-using-KNN/blob/main/output/AboutDiabetes.jpg?raw=true" width="100%">
</div>

## Objectives:
In this project our goal is to Predict the onset of diabetes based on diagnostic measures.
## Dataset:
[Pima Indians Diabetes Database](https://www.kaggle.com/uciml/pima-indians-diabetes-database)

### About Data:
This dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. The objective of the dataset is to diagnostically predict whether or not a patient has diabetes, based on certain diagnostic measurements included in the dataset. Several constraints were placed on the selection of these instances from a larger database. In particular, all patients here are females at least 21 years old of Pima Indian heritage.<br>
The datasets consists of several medical predictor variables and one target variable, Outcome. Predictor variables includes the number of pregnancies the patient has had, their BMI, insulin level, age, and so on.
## Implementation:

**Libraries:** `sklearn` `Matplotlib` `pandas` `seaborn` `NumPy` `Scipy` 


## A few glimpses of EDA:
### Features of dataset:
![Features1](https://github.com/Pradnya1208/Diabetes-classification-using-KNN/blob/main/output/eda1.PNG?raw=true)
![Features2](https://github.com/Pradnya1208/Diabetes-classification-using-KNN/blob/main/output/eda2.PNG?raw=true)


## Data Imputation:
```
diabetes_data_copy['Glucose'].fillna(diabetes_data_copy['Glucose'].mean(), inplace = True)
diabetes_data_copy['BloodPressure'].fillna(diabetes_data_copy['BloodPressure'].mean(), inplace = True)
diabetes_data_copy['SkinThickness'].fillna(diabetes_data_copy['SkinThickness'].median(), inplace = True)
diabetes_data_copy['Insulin'].fillna(diabetes_data_copy['Insulin'].median(), inplace = True)
diabetes_data_copy['BMI'].fillna(diabetes_data_copy['BMI'].median(), inplace = True)
```
### Plotting feartures after imputation:
![Imputed data1](https://github.com/Pradnya1208/Diabetes-classification-using-KNN/blob/main/output/eda_nan1.PNG?raw=true)
![Imputed data2](https://github.com/Pradnya1208/Diabetes-classification-using-KNN/blob/main/output/eda_nan2.PNG?raw=true)



## Model Training and Evaluation:

### KNN
```
for i in range(1,15):

    knn = KNeighborsClassifier(i)
    knn.fit(X_train,y_train)
    
```
```
Max test score 76.5625 % and k = [11]
```
![Result](https://github.com/Pradnya1208/Diabetes-classification-using-KNN/blob/main/output/traintestscore.PNG?raw=true)

### Plotting Decision Regions:
![Decision Regions](https://github.com/Pradnya1208/Diabetes-classification-using-KNN/blob/main/output/Decision%20regions.PNG?raw=true)


### Confusion Matrix:
The confusion matrix is a technique used for summarizing the performance of a classification algorithm i.e. it has binary outputs.
![CM](https://github.com/Pradnya1208/Diabetes-classification-using-KNN/blob/main/output/confusion%20matrix.PNG?raw=true)<br>

**Results:**<br>
<img src = "https://github.com/Pradnya1208/Diabetes-classification-using-KNN/blob/main/output/cMresults.PNG?raw=true">   
<img src = "https://github.com/Pradnya1208/Diabetes-classification-using-KNN/blob/main/output/plotcm.PNG?raw=true">

### Classification Report:
> **Precision** is the ratio of correctly predicted positive observations to the total predicted positive observations. The question that this metric answer is of all passengers that labeled as survived, how many actually survived? High precision relates to the low false positive rate. We have got 0.788 precision which is pretty good.

Precision = TP/TP+FP

> **Recall** (Sensitivity) is the ratio of correctly predicted positive observations to the all observations in actual class - yes. The question recall answers is: Of all the passengers that truly survived, how many did we label? A recall greater than 0.5 is good.

Recall = TP/TP+FN

> **F1 Score** is the weighted average of Precision and Recall. Therefore, this score takes both false positives and false negatives into account. Intuitively it is not as easy to understand as accuracy, but F1 is usually more useful than accuracy, especially if you have an uneven class distribution. Accuracy works best if false positives and false negatives have similar cost. If the cost of false positives and false negatives are very different, it’s better to look at both Precision and Recall.

F1 Score = 2(Recall Precision) / (Recall + Precision)

![classificationreport](https://github.com/Pradnya1208/Diabetes-classification-using-KNN/blob/main/output/classification%20report.PNG?raw=true)

### ROC-AUC:
ROC (Receiver Operating Characteristic) Curve tells us about how good the model can distinguish between two things (e.g If a patient has a disease or no). Better models can accurately distinguish between the two. 
Whereas, a poor model will have difficulties in distinguishing between the two.<br>
![rocauc](https://github.com/Pradnya1208/Diabetes-classification-using-KNN/blob/main/output/rocauc.PNG?raw=true)



## Optimizations

* Scaling:
It is always advisable to bring all the features to the same scale for applying distance based algorithms like KNN.<br>
We can imagine how the feature with greater range with overshadow or dimenish the smaller feature completely and this will impact the performance of all distance based model as it will give higher weightage to variables which have higher magnitude.

* Cross Validation:
When model is split into training and testing it can be possible that specific type of data point may go entirely into either training or testing portion. This would lead the model to perform poorly. Hence over-fitting and underfitting problems can be well avoided with cross validation techniques.
**Stratify** parameter makes a split so that the proportion of values in the sample produced will be the same as the proportion of values provided to parameter stratify.

For example, if variable y is a binary categorical variable with values 0 and 1 and there are 25% of zeros and 75% of ones, stratify=y will make sure that your random split has 25% of 0's and 75% of 1's.

![Crossvalidation](https://github.com/Pradnya1208/Diabetes-classification-using-KNN/blob/main/output/crossvalidation.PNG?raw=true)

* Hyperparameter Tuning:

Grid search is an approach to hyperparameter tuning that will methodically build and evaluate a model for each combination of algorithm parameters specified in a grid.
```
from sklearn.model_selection import GridSearchCV
parameters_grid = {"n_neighbors": np.arange(0,50)}
knn= KNeighborsClassifier()
knn_GSV = GridSearchCV(knn, param_grid=parameters_grid, cv = 5)
knn_GSV.fit(X, y)
print("Best Params" ,knn_GSV.best_params_)
print("Best score" ,knn_GSV.best_score_)
```
```
Best Params {'n_neighbors': 25}
Best score 0.7721840251252015
```

### Lessons Learned

`Data Imputation`
`Handling Outliers`
`Feature Engineering`
`Classification Models`
`Parameter Optimization`
## References:
[Skewness](https://www.statisticshowto.datasciencecentral.com/probability-and-statistics/skewed-distribution/)
[Scaling](https://medium.com/@rrfd/standardize-or-normalize-examples-in-python-e3f174b65dfc)
[Rescaling the data in ML using scikit lerarn](https://machinelearningmastery.com/rescaling-data-for-machine-learning-in-python-with-scikit-learn/)
[Confusion Matrix](https://medium.com/@djocz/confusion-matrix-aint-that-confusing-d29e18403327)
[Classification Report](http://joshlawman.com/metrics-classification-report-breakdown-precision-recall-f1/)

### Feedback

If you have any feedback, please reach out at pradnyapatil671@gmail.com


### 🚀 About Me
#### Hi, I'm Pradnya! 👋
I am an AI Enthusiast and  Data science & ML practitioner

[1]: https://github.com/Pradnya1208
[2]: https://www.linkedin.com/in/pradnya-patil-b049161ba/
[3]: https://public.tableau.com/app/profile/pradnya.patil3254#!/
[4]: https://twitter.com/Pradnya1208


[![github](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/c292abd3f9cc647a7edc0061193f1523e9c05e1f/icons/git.svg)][1]
[![linkedin](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/9f5c4a255972275ced549ea6e34ef35019166944/icons/iconmonstr-linkedin-5.svg)][2]
[![tableau](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/e257c5d6cf02f13072429935b0828525c601414f/icons/icons8-tableau-software%20(1).svg)][3]
[![twitter](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/c9f9c5dc4e24eff0143b3056708d24650cbccdde/icons/iconmonstr-twitter-5.svg)][4]

