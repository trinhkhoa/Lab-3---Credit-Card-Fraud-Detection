# Lab-3---Credit-Card-Fraud-Detection
![image](https://user-images.githubusercontent.com/77667121/222990961-86e3395c-11c2-4347-ac9e-09da09fbdff0.png)

**This Lab is due Wednesday night, February 8.**

Please use this .ipynb file as a starting point and write your code and fill in your answers directly in the notebook. I will ask specific questions in bolded font . For written answers, please create the cell as a Markdown type so the text displays neatly. For code, comment sufficiently so I can understand your process. Please try to delete unnecessary, old code so it is easy for me to evluate. When you have finished, displayed all code outputs and written all answers, please save and submit to Canvas as a .html file.

This is a group lab - you may work together with one partner and submit one solution if you choose. If you do, be sure to include both names and please include a paragraph detailing the contributions of each partner.

## Background and Objective:
Credit card fraud is a huge and growing problem worldwide. Fraud losses affect card issuers, merchants, and consumers alike. The Federal Trade Commission reported $181 miliion lost across 389,737 cases of credit card fraud in 2021 - an 18% increase from 2020 alone! For more information and a breakdown of the types of fraud, read the FTC report at https://www.ftc.gov/system/files/ftc_gov/pdf/CSN%20Annual%20Data%20Book%202021%20Final%20PDF.pdf.

Machine learning methods underly the automated systems that detect and halt fraudulent card use. A major challenge faced by banks is how to automatically detect fraudulent transactions in real time and halt them before the purchase can be completed. When a fraudulent use is suspected, typically the transaction will be put on hold and a confirmation sent to the consumer in the form of a text or phone call. In this lab we will review methods from last week's lab for creating and evaluating predictions and use a new tool in Pareto optimal frontiers for making decisions.

## Data:
The data 'creditcard.csv' contains the data we will use. 28 columns of variables are provided that are similar to what is known as PCA (Principle Component Analysis). You may have seen this in a different section of DA 350 or a DA 220 - if not, all you need to know is they are transformations of original columns created to be information-dense but do not have clear definitions. This is reflected in the variable names, which you won't recognize as anything interpretable. One advantage to this is privacy, as the data origin was able to provide real financial records without comprimising anyone's confidentiality. In addition to these 28 columns, the data also contains the amount of the transaction and Fraud indicating 0 if the transaction is valid, and 1 if the transaction is fraudulent.

```python
#Import packages
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics
from sklearn.model_selection import train_test_split
```

**1) Start by reading in the data and examining the Fraud outcome variable. Normalize the Amount column so it has mean 0 and standard deviation 1.**

0 is a valid transaction, and 1 is a fraudulent transaction. All columns except Amount and Fraud have each been normalized to have mean 0, so they can be compared to each other in an unbiased way. Recall that you normalize a variable by subtracting the mean and dividing by the standard deviation.

```python
df = pd.read_csv('creditcard-1.csv')
mean = df["Amount"].mean()
std = df["Amount"].std()
df["Amount"] = (df["Amount"] - df["Amount"].mean()) / df["Amount"].std()
```
Now we will explore some visualizations to investigate what predictors may be relevant for detecting fraud. Before plugging everything into a complicated algorithm, it’s good to have some sense of simple visuals confirming their validity. You’re specifically looking for predictor variables that have a much different distribution in valid and fraudulent usage.

**2) Create a few visuals here and explain what they tell you. Choose a few columns based off of these visuals to use in your predictive models below.**
```python
grouping_column = "Fraud"

for column in df.columns:
    if column == grouping_column:
        continue
    ax = df.boxplot(column, by=grouping_column)
    ax.set_ylim(-10, 10)
    plt.tight_layout()
    plt.show()
```
We are looking for columns with reasonably different distribution between types Fraud = 0 and 1. This will allow for clearer distinction and prediction of which transactions would be fraudulent. We used box plots to summarize the median, mean, upper and lower quartile of each column, divided by Fraud = 0 and 1. Going through every boxplot, we decided to use the columns below:
```python
df_pred = df.loc[:, ["V1", "V2", "V3", "V4", "V6", "V9", "V10", "V11", "V12", "V14", "V16", "V17", "Amount", "Fraud"]]
```
Often in real world projects we have only a single data set and don't have a separate training and test set. In such cases, it is standard practice to split the data into a training set and a test set to better estimate future model performance.

**3) Split your selected columns of data 70/30 into training and test sets using the 'train_test_split' function of sklearn, which you can read more about here: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html. Set the 'random state' to your favorite integer before doing so to ensure you will have consistent training and test sets if you rerun the code.**
```python
df = pd.DataFrame(df_pred)
nFraud = df.drop("Fraud", axis='columns')
X_train, X_test, y_train, y_test = train_test_split(nFraud, df['Fraud'], test_size=0.3, random_state=11)
```
**4) Run a k-nearest-Neighbors classifier with default k value and delta to predict whether a transaction is fradulent based on the predictors. Examine the performance metrics and produce your confusion matrix - how well is the model doing?**
```python
knn = KNeighborsClassifier() 
knn.fit(X_train, y_train) 
class_predictions = knn.predict(X_train)
print("Accuracy:", sklearn.metrics.accuracy_score(y_train, class_predictions))
confusion_matrix = sklearn.metrics.confusion_matrix(y_train, class_predictions)
cm_display = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
cm_display.plot()
plt.show()
print(sklearn.metrics.classification_report(y_train, class_predictions))
```
```
Accuracy: 0.9995585963363496
```
The overall accuracy is 100%. The precision and recall rate for class 0 are both 100%. The precision and recall rate for class 1 is slightly worse, at 92% and 81% each respectively. This is still really good predictions, however, it shows that we might be overfitting the dataset

**5) Run a series of models with a range of values of k and all the delta possibilities within each. Record the TP, FP, TN, and FN in a table for each option.**

Note that the computational time might be long - be smart about computational time and plan accordingly!
```python
k_values = []
for num in range(1, 30):
    if num % 2 != 0:
        k_values.append(num)
table = pd.DataFrame()
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k, 
                                      p=2,    # Euclidian
                                      metric="minkowski")
    knn.fit(X_train, y_train)
    prediction = knn.predict_proba(X_train)
    perf = []
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_train, prediction[:,1])
    for delta in thresholds:
        TP = np.sum(np.logical_and(prediction[:,1] >= delta, y_train == 1))
        FP = np.sum(np.logical_and(prediction[:,1] >= delta, y_train == 0))
        TN = np.sum(np.logical_and(prediction[:,1] < delta, y_train == 0))
        FN = np.sum(np.logical_and(prediction[:,1] < delta, y_train == 1))
        FL = 500 * FN
        perf.append([k,delta, TP, FP, TN, FN, FL])
    perf_table = pd.DataFrame(perf, columns = ['k','Delta', 'True Positives', 'False Positives', 'True Negatives', 'False Negatives','Fraud Loss'])
    table = pd.concat([table, perf_table])
```

Unlike last week's lab with the movie ads, we do not have directly comparable costs and benefits in this situation. Failing to identify and stop a fraudulent transaction costs our bank quantifiable dollar amounts, while unnecessarily halting a transaction results in customer frustration and ill-will that is not easily quantifiable.

This is a perfect case to construct a Pareto optimal curve as we learned in class to balance the two incomparable costs.

For the financial costs of failing to halt a fraudulent transaction, use the median fraud losses in Ohio from the FTC guide in the introduction. If you were working for a bank, you might look to your own records for this number, but a public sourced number could also be a solid estimate.

**6) Start by plotting all of your solutions from 5), with the financial costs of undetected fraud on the x-axis, and the number of valid transactions erroneously stopped on the y-axis.**
```python
a = np.array(table['Fraud Loss'])
b = np.array(table['False Positives'])
# Add x and y labels and a title
plt.scatter(a, b)
plt.xlabel("Financial costs of undetected fraud")
plt.ylabel("Number of valid transactions stopped")

# Show the legend
plt.legend()

# Show the plot
plt.show()
```

**7) Now improve on your answer from 6) by reducing to only solutions on the Pareto optimal frontier - that is, solutions that are not strictly dominated in both directions by another solution. Display the table and graph the Pareto optimal solutions.**

While there may exist some obscure packages and functions that exist to do this computation for you, it is a fairly simple exercise to code. Code it yourself to fully appreciate how the optimal frontier is constructed. You should be able to accomplish the task with 2 nested for-loops.

```python
def pareto(df):
  frontier = []
  value = []
  df = df.sort_values(by = ["Fraud Loss", "False Positives"], ascending = [True, False], na_position = "first")
  df.drop_duplicates(subset="Fraud Loss", keep ="last", inplace = True)
  df = df.sort_values(by = ["Fraud Loss", "False Positives"], ascending = [True, False], na_position = "first")
  df.drop_duplicates(subset="False Positives", keep ="first", inplace = True)
  return df
```
```python
frontier = pareto(table)
frontier_df = pd.DataFrame(frontier)
```
```python
a = np.array(frontier_df['Fraud Loss'])
b = np.array(frontier_df['False Positives'])

# Add x and y labels and a title
plt.scatter(a, b)
plt.xlabel("Financial costs of undetected fraud")
plt.ylabel("Number of valid transactions stopped")

plt.show()
```
Finally, envision that the full data set represents one week of transactions (and therefore the training data represents 7/10 of one week of transactions.)

**8a) Your boss says she wants to lose no more than $1,500,000 a year in fraudulent transactions. From the training data, what decision should you make about which model/parameters to use, and what do you tell your boss about the expected number of valud transactions you will halt over the course of the year?**

**8b) Using your model from 8a), evaluate your predictions on the test set and extrapolate how much loss to fraudulent transactions and how many unnecessarily stopped valid transactions occur over the next year.**

**8c) Conversely, your boss is now worried about customer reputation and wants to errorenously halt no more than 1000 valid transactions over the course of the year - what decision should you make about which model/parameters to use, and what do you tell your boss about the expected loss to undetected fraudulent transactions over the course of the year?**

**8d) Using your model from 8c), evaluate your predictions on the test set and extrapolate how much loss to fraudulent transactions and how many unnecessarily stopped valid transactions occur over the next year.**

Note that in both cases, the actual performance (evaluated from the test set) over the next year may be off of the marginal targets by a bit - for example, you may end up with 1,050 valid transactions halted. Slight differences should be expected because of the variability in training and test set peerformance, however large deviations should give you pause.
```python
frontier_df["Fraud Loss"] = frontier_df["Fraud Loss"] * 52/0.7 
frontier_df["False Positives"] = frontier_df["False Positives"] * 52/0.7
frontier_df["True Positives"] = frontier_df["True Positives"] * 52/0.7
frontier_df["True Negatives"] = frontier_df["True Negatives"] * 52/0.7
frontier_df["False Negatives"] = frontier_df["False Negatives"] * 52/0.7
```
```
    k	Delta	    True Positives	False Positives	True Negatives	False Negatives	Fraud Loss
22	27	0.074074	21914.285714	9211.428571	1.477595e+07	2822.857143	1.411429e+06
21	23	0.086957	21840.000000	8542.857143	1.477662e+07	2897.142857	1.448571e+06
16	17	0.117647	21765.714286	7725.714286	1.477743e+07	2971.428571	1.485714e+06
14	15	0.133333	21691.428571	7428.571429	1.477773e+07	3045.714286	1.522857e+06
12	13	0.153846	21542.857143	6908.571429	1.477825e+07	3194.285714	1.597143e+06
10	11	0.181818	21468.571429	5942.857143	1.477922e+07	3268.571429	1.634286e+06
8	9	0.222222	21394.285714	4828.571429	1.478033e+07	3342.857143	1.671429e+06
6	7	0.285714	21320.000000	4160.000000	1.478100e+07	3417.142857	1.708571e+06
4	5	0.400000	21022.857143	3417.142857	1.478174e+07	3714.285714	1.857143e+06
7	9	0.333333	20948.571429	3491.428571	1.478167e+07	3788.571429	1.894286e+06
13	17	0.294118	20800.000000	4382.857143	1.478078e+07	3937.142857	1.968571e+06
5	7	0.428571	20725.714286	2971.428571	1.478219e+07	4011.428571	2.005714e+06
6	9	0.444444	20577.142857	3120.000000	1.478204e+07	4160.000000	2.080000e+06
2	3	0.666667	20502.857143	668.571429	1.478449e+07	4234.285714	2.117143e+06
16	21	0.285714	20428.571429	4085.714286	1.478107e+07	4308.571429	2.154286e+06
14	19	0.315789	20354.285714	3937.142857	1.478122e+07	4382.857143	2.191429e+06
12	17	0.352941	20280.000000	3862.857143	1.478130e+07	4457.142857	2.228571e+06
10	15	0.400000	20205.714286	3342.857143	1.478182e+07	4531.428571	2.265714e+06
17	29	0.241379	20131.428571	4457.142857	1.478070e+07	4605.714286	2.302857e+06
5	9	0.555556	20057.142857	2154.285714	1.478301e+07	4680.000000	2.340000e+06
3	5	0.600000	19982.857143	1782.857143	1.478338e+07	4754.285714	2.377143e+06
6	11	0.545455	19834.285714	2302.857143	1.478286e+07	4902.857143	2.451429e+06
10	29	0.689655	19685.714286	2228.571429	1.478293e+07	5051.428571	2.525714e+06
3	7	0.714286	19462.857143	1262.857143	1.478390e+07	5274.285714	2.637143e+06
5	13	0.692308	19388.571429	1708.571429	1.478345e+07	5348.571429	2.674286e+06
7	21	0.714286	19314.285714	1857.142857	1.478330e+07	5422.857143	2.711429e+06
2	5	0.800000	19091.428571	371.428571	1.478479e+07	5645.714286	2.822857e+06
5	15	0.733333	19017.142857	1485.714286	1.478367e+07	5720.000000	2.860000e+06
4	11	0.727273	18942.857143	1411.428571	1.478375e+07	5794.285714	2.897143e+06
1	3	1.000000	18422.857143	0.000000	1.478516e+07	6314.285714	3.157143e+06
3	9	0.777778	18200.000000	965.714286	1.478419e+07	6537.142857	3.268571e+06
7	29	0.793103	17977.142857	1560.000000	1.478360e+07	6760.000000	3.380000e+06
5	21	0.809524	17828.571429	1188.571429	1.478397e+07	6908.571429	3.454286e+06
3	11	0.818182	17382.857143	817.142857	1.478434e+07	7354.285714	3.677143e+06
5	23	0.826087	17160.000000	1114.285714	1.478405e+07	7577.142857	3.788571e+06
3	13	0.846154	16714.285714	742.857143	1.478442e+07	8022.857143	4.011429e+06
4	19	0.842105	16640.000000	891.428571	1.478427e+07	8097.142857	4.048571e+06
3	15	0.866667	15748.571429	594.285714	1.478457e+07	8988.571429	4.494286e+06
2	9	0.888889	15377.142857	297.142857	1.478486e+07	9360.000000	4.680000e+06
2	11	0.909091	13594.285714	222.857143	1.478494e+07	11142.857143	5.571429e+06
4	29	0.896552	10325.714286	520.000000	1.478464e+07	14411.428571	7.205714e+06
2	17	0.941176	9360.000000	74.285714	1.478509e+07	15377.142857	7.688571e+06
3	27	0.925926	8097.142857	148.571429	1.478501e+07	16640.000000	8.320000e+06
```
**8a**
```python
op_df = frontier_df.loc[frontier_df["Fraud Loss"] < 1500000]
op_df
```
```
	k	Delta	    True Positives	False Positives	True Negatives	False Negatives	Fraud Loss
22	27	0.074074	21914.285714	9211.428571	1.477595e+07	2822.857143	1.411429e+06
21	23	0.086957	21840.000000	8542.857143	1.477662e+07	2897.142857	1.448571e+06
16	17	0.117647	21765.714286	7725.714286	1.477743e+07	2971.428571	1.485714e+06
```
**8b**
```python
for k in op_df.loc[op_df["False Positives"] > 0]["k"]: #choose params that give stopped valid transactions greater than 0
  for delta in op_df.loc[(op_df["k"] == k)]["Delta"]:
    knn = KNeighborsClassifier(n_neighbors = k,
                                  p=2,
                                  metric="minkowski")
    knn.fit(X_train, y_train)
    prob_predictions = knn.predict_proba(X_test)
    class_predictions = []
    for i in range(len(prob_predictions)):
      if prob_predictions[i,1] >= delta:
        class_predictions.append(1)
      else:
        class_predictions.append(0)
    print(k, delta)
    print(sklearn.metrics.classification_report(y_test, class_predictions))
```
```
precision    recall  f1-score   support

           0       1.00      1.00      1.00     85284
           1       0.74      0.82      0.78       159

    accuracy                           1.00     85443
   macro avg       0.87      0.91      0.89     85443
weighted avg       1.00      1.00      1.00     85443
```
```python
perf_table
```
```
    k	Delta	    True Positives	False Positives	True Negatives	False Negatives	Fraud Loss
0	17	0.117647	9731.428571	3491.428571	6.331891e+06	2080.0	1040000.0
```
Using our model, we would stop 3491 valid transactions and experience about 1.04 million in Fraud Loss which is way below the target 1.5 million so our model is great *
**8c**
```python
fp_df = frontier_df.loc[frontier_df["False Positives"] < 1000]
fp_df
```
```
    k	Delta	    True Positives	False Positives	True Negatives	False Negatives	Fraud Loss
2	3	0.666667	20502.857143	668.571429	1.478449e+07	4234.285714	2.117143e+06
2	5	0.800000	19091.428571	371.428571	1.478479e+07	5645.714286	2.822857e+06
1	3	1.000000	18422.857143	0.000000	1.478516e+07	6314.285714	3.157143e+06
3	9	0.777778	18200.000000	965.714286	1.478419e+07	6537.142857	3.268571e+06
3	11	0.818182	17382.857143	817.142857	1.478434e+07	7354.285714	3.677143e+06
3	13	0.846154	16714.285714	742.857143	1.478442e+07	8022.857143	4.011429e+06
4	19	0.842105	16640.000000	891.428571	1.478427e+07	8097.142857	4.048571e+06
3	15	0.866667	15748.571429	594.285714	1.478457e+07	8988.571429	4.494286e+06
2	9	0.888889	15377.142857	297.142857	1.478486e+07	9360.000000	4.680000e+06
2	11	0.909091	13594.285714	222.857143	1.478494e+07	11142.857143	5.571429e+06
4	29	0.896552	10325.714286	520.000000	1.478464e+07	14411.428571	7.205714e+06
2	17	0.941176	9360.000000	74.285714	1.478509e+07	15377.142857	7.688571e+06
3	27	0.925926	8097.142857	148.571429	1.478501e+07	16640.000000	8.320000e+06
```
**8d**
```python
k = 11
delta = 0.818182
knn = KNeighborsClassifier(n_neighbors = k,
                                p=2,
                                metric="minkowski")
knn.fit(X_train, y_train)
prob_predictions = knn.predict_proba(X_test)
class_predictions = []
perf = []
for i in range(len(prob_predictions)):
    if prob_predictions[i,1] >= delta:
        class_predictions.append(1)
    else:
        class_predictions.append(0)
TP = np.sum(np.logical_and(prob_predictions[:,1] >= delta, y_test == 1)) * 52/0.7  
FP = np.sum(np.logical_and(prob_predictions[:,1] >= delta, y_test == 0)) * 52/0.7 
TN = np.sum(np.logical_and(prob_predictions[:,1] < delta, y_test == 0)) * 52/0.7 
FN = np.sum(np.logical_and(prob_predictions[:,1] < delta, y_test == 1)) * 52/0.7 
FL = 500 * FN  
perf.append([k,delta, TP, FP, TN, FN, FL])
perf_table = pd.DataFrame(perf, columns = ['k','Delta', 'True Positives', 'False Positives', 'True Negatives', 'False Negatives','Fraud Loss'])
print(sklearn.metrics.classification_report(y_test, class_predictions))
```
```
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     85284
           1       0.98      0.55      0.70       159

    accuracy                           1.00     85443
   macro avg       0.99      0.77      0.85     85443
weighted avg       1.00      1.00      1.00     85443
```
```python
perf_table
```
```
    k	Delta	    True Positives	False Positives	True Negatives	False Negatives	Fraud Loss
0	11	0.818182	6462.857143	148.571429	6.335234e+06	5348.571429	2.674286e+06
```
Using our model, we would have about 148 - 149 stopped valid transactions and about 2.67 million in Fraud Loss. *
