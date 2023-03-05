# Lab-3---Credit-Card-Fraud-Detection
![image](https://user-images.githubusercontent.com/77667121/222990961-86e3395c-11c2-4347-ac9e-09da09fbdff0.png)
This Lab is due Wednesday night, February 8.
Please use this .ipynb file as a starting point and write your code and fill in your answers directly in the notebook. I will ask specific questions in bolded font . For written answers, please create the cell as a Markdown type so the text displays neatly. For code, comment sufficiently so I can understand your process. Please try to delete unnecessary, old code so it is easy for me to evluate. When you have finished, displayed all code outputs and written all answers, please save and submit to Canvas as a .html file.

This is a group lab - you may work together with one partner and submit one solution if you choose. If you do, be sure to include both names and please include a paragraph detailing the contributions of each partner.

Background and Objective:
Credit card fraud is a huge and growing problem worldwide. Fraud losses affect card issuers, merchants, and consumers alike. The Federal Trade Commission reported $181 miliion lost across 389,737 cases of credit card fraud in 2021 - an 18% increase from 2020 alone! For more information and a breakdown of the types of fraud, read the FTC report at https://www.ftc.gov/system/files/ftc_gov/pdf/CSN%20Annual%20Data%20Book%202021%20Final%20PDF.pdf.

Machine learning methods underly the automated systems that detect and halt fraudulent card use. A major challenge faced by banks is how to automatically detect fraudulent transactions in real time and halt them before the purchase can be completed. When a fraudulent use is suspected, typically the transaction will be put on hold and a confirmation sent to the consumer in the form of a text or phone call. In this lab we will review methods from last week's lab for creating and evaluating predictions and use a new tool in Pareto optimal frontiers for making decisions.

Data:
The data 'creditcard.csv' contains the data we will use. 28 columns of variables are provided that are similar to what is known as PCA (Principle Component Analysis). You may have seen this in a different section of DA 350 or a DA 220 - if not, all you need to know is they are transformations of original columns created to be information-dense but do not have clear definitions. This is reflected in the variable names, which you won't recognize as anything interpretable. One advantage to this is privacy, as the data origin was able to provide real financial records without comprimising anyone's confidentiality. In addition to these 28 columns, the data also contains the amount of the transaction and Fraud indicating 0 if the transaction is valid, and 1 if the transaction is fraudulent.

[ ]
#Import packages
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics
from sklearn.model_selection import train_test_split
[ ]
df = pd.read_csv('creditcard-1.csv')
mean = df["Amount"].mean()
std = df["Amount"].std()
df["Amount"] = (df["Amount"] - df["Amount"].mean()) / df["Amount"].std()
1) Start by reading in the data and examining the Fraud outcome variable. Normalize the Amount column so it has mean 0 and standard deviation 1.

0 is a valid transaction, and 1 is a fraudulent transaction. All columns except Amount and Fraud have each been normalized to have mean 0, so they can be compared to each other in an unbiased way. Recall that you normalize a variable by subtracting the mean and dividing by the standard deviation.

Now we will explore some visualizations to investigate what predictors may be relevant for detecting fraud. Before plugging everything into a complicated algorithm, it’s good to have some sense of simple visuals confirming their validity. You’re specifically looking for predictor variables that have a much different distribution in valid and fraudulent usage.

2) Create a few visuals here and explain what they tell you. Choose a few columns based off of these visuals to use in your predictive models below.

[ ]
grouping_column = "Fraud"

for column in df.columns:
    if column == grouping_column:
        continue
    ax = df.boxplot(column, by=grouping_column)
    ax.set_ylim(-10, 10)
    plt.tight_layout()
    plt.show()
    
  # loại: 13, 15, 22 - 28, Amount, 8(?), 19(?), 20(?), 21(?)
  # dùng: 1, 2, 3, 4, 5(?), 6, 8(?), 9, 10, 11, 12, 14, 16, 17, 18(?), 19(?), 21(?),  

[ ]
df_pred = df.loc[:, ["V1", "V2", "V3", "V4", "V6", "V9", "V10", "V11", "V12", "V14", "V16", "V17", "Amount", "Fraud"]]
Often in real world projects we have only a single data set and don't have a separate training and test set. In such cases, it is standard practice to split the data into a training set and a test set to better estimate future model performance.

3) Split your selected columns of data 70/30 into training and test sets using the 'train_test_split' function of sklearn, which you can read more about here: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html. Set the 'random state' to your favorite integer before doing so to ensure you will have consistent training and test sets if you rerun the code.

[ ]
df = pd.DataFrame(df_pred)
nFraud = df.drop("Fraud", axis='columns')
X_train, X_test, y_train, y_test = train_test_split(nFraud, df['Fraud'], test_size=0.3, random_state=11)
4) Run a k-nearest-Neighbors classifier with default k value and delta to predict whether a transaction is fradulent based on the predictors. Examine the performance metrics and produce your confusion matrix - how well is the model doing?

[ ]
knn = KNeighborsClassifier() 
knn.fit(X_train, y_train) 
class_predictions = knn.predict(X_train)
print("Accuracy:", sklearn.metrics.accuracy_score(y_train, class_predictions))
confusion_matrix = sklearn.metrics.confusion_matrix(y_train, class_predictions)
cm_display = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
cm_display.plot()
plt.show()
print(sklearn.metrics.classification_report(y_train, class_predictions))

Nhấp đúp (hoặc nhấn Enter) để chỉnh sửa

5) Run a series of models with a range of values of k and all the delta possibilities within each. Record the TP, FP, TN, and FN in a table for each option.

Note that the computational time might be long - be smart about computational time and plan accordingly!

[ ]
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
[ ]
table

Unlike last week's lab with the movie ads, we do not have directly comparable costs and benefits in this situation. Failing to identify and stop a fraudulent transaction costs our bank quantifiable dollar amounts, while unnecessarily halting a transaction results in customer frustration and ill-will that is not easily quantifiable.

This is a perfect case to construct a Pareto optimal curve as we learned in class to balance the two incomparable costs.

For the financial costs of failing to halt a fraudulent transaction, use the median fraud losses in Ohio from the FTC guide in the introduction. If you were working for a bank, you might look to your own records for this number, but a public sourced number could also be a solid estimate.

6) Start by plotting all of your solutions from 5), with the financial costs of undetected fraud on the x-axis, and the number of valid transactions erroneously stopped on the y-axis.

[ ]
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


### 7) Now improve on your answer from 6) by reducing to only solutions on the Pareto optimal frontier - that is, solutions that are not strictly dominated in both directions by another solution. Display the table and graph the Pareto optimal solutions.

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

frontier = pareto(table)
frontier_df = pd.DataFrame(frontier)
a = np.array(frontier_df['Fraud Loss'])
b = np.array(frontier_df['False Positives'])

# Add x and y labels and a title
plt.scatter(a, b)
plt.xlabel("Financial costs of undetected fraud")
plt.ylabel("Number of valid transactions stopped")

plt.show()
```
Finally, envision that the full data set represents one week of transactions (and therefore the training data represents 7/10 of one week of transactions.)

### 8a) Your boss says she wants to lose no more than $1,500,000 a year in fraudulent transactions. From the training data, what decision should you make about which model/parameters to use, and what do you tell your boss about the expected number of valud transactions you will halt over the course of the year?

### 8b) Using your model from 8a), evaluate your predictions on the test set and extrapolate how much loss to fraudulent transactions and how many unnecessarily stopped valid transactions occur over the next year.

### 8c) Conversely, your boss is now worried about customer reputation and wants to errorenously halt no more than 1000 valid transactions over the course of the year - what decision should you make about which model/parameters to use, and what do you tell your boss about the expected loss to undetected fraudulent transactions over the course of the year?

### 8d) Using your model from 8c), evaluate your predictions on the test set and extrapolate how much loss to fraudulent transactions and how many unnecessarily stopped valid transactions occur over the next year.

Note that in both cases, the actual performance (evaluated from the test set) over the next year may be off of the marginal targets by a bit - for example, you may end up with 1,050 valid transactions halted. Slight differences should be expected because of the variability in training and test set peerformance, however large deviations should give you pause.
```python
frontier_df["Fraud Loss"] = frontier_df["Fraud Loss"] * 52/0.7 
frontier_df["False Positives"] = frontier_df["False Positives"] * 52/0.7
frontier_df["True Positives"] = frontier_df["True Positives"] * 52/0.7
frontier_df["True Negatives"] = frontier_df["True Negatives"] * 52/0.7
frontier_df["False Negatives"] = frontier_df["False Negatives"] * 52/0.7
```
### 8a
```python
op_df = frontier_df.loc[frontier_df["Fraud Loss"] < 1500000]
op_df
```
### 8b
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
### 8d
```python
fp_df = frontier_df.loc[frontier_df["False Positives"] < 1000]
for k in fp_df.loc[fp_df["False Positives"] > 0]["k"]: #choose params that give stopped valid transactions greater than 0
  for delta in fp_df.loc[(fp_df["k"] == k)]["Delta"]:
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
