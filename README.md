# Logistic Regression model - Home Equity dataset (HMEQ)

### 📌 Project Overview
---
This is a project from Kaggle to explore a dataset concerning the loan status on a home equity line of credit. (https://www.kaggle.com/datasets/ajay1735/hmeq-data)

**Context:** The consumer credit department of a bank wants to automate the decision making process for approval of home equity lines of credit. To do this, they will follow the recommendations of the Equal Credit Opportunity Act to create an empirically derived and statistically sound credit scoring model. The model will be based on data collected from recent applicants granted credit through the current process of loan underwriting. The model will be built from predictive modeling tools, but the created model must be sufficiently interpretable to provide a reason for any adverse actions (rejections).
### 📊 Dataset Overview
---
The Home Equity dataset (HMEQ) contains baseline and loan performance information for 5,960 recent home equity loans. The target (BAD) is a binary variable indicating whether an applicant eventually defaulted or was seriously delinquent. This adverse outcome occurred in 1,189 cases (20%). For each applicant, 12 input variables were recorded.

**Home Equity Data Dictionary**

| Variables | Role | Description |
|-----------|------| ------------|
| **BAD**  | Response | `1` = Client defaulted on loan, `0` = Loan repaid |
| **LOAN** | Predictor | Amount of the loan request |
| **MORTDUE** | Predictor | Amount due on existing mortgage |
| **VALUE** | Predictor | Value of current property |
| **REASON** | Predictor | `DebtCon` = Debt consolidation, `HomeImp` = Home improvement |
| **JOB** | Predictor | Six occupational categories |
| **YOJ** | Predictor | Years at present job |
| **DEROG** | Predictor | Number of major derogatory reports |
| **DELINQ** | Predictor | Number of delinquent credit lines |
| **CLAGE** | Predictor | Age of oldest trade line in months |
| **NINQ** | Predictor | Number of recent credit lines |
| **CLNO** | Predictor | Number of credit lines |
| **DEBTINC** | Predictor | Debt-to-income ratio |

### 🚀 Project Goals
---
✔ Data Cleaning & Exploration: Handle missing values, detect outliers, explore distribution, and visualize key trends.

✔ Feature Engineering: Transform variables for better model performance.

✔ Predictive Modeling: Apply Logistic Regression model to assess loan risk.

✔ Interpretability: Ensure the model provides understandable explanations for loan decisions and calculate Credit Score for dataset.

### 📂 Files
---
Data Sample/hmeq.csv : Raw dataset 

Script/Logistic Regression model.ipynb : Exploratory Data Analysis (EDA) and modeling

README.md : Project documentation

### 💡Requirement
---
Platform: Jupyter Notebook - Python

Packages/Setup:
```
### set up environment 
import os
import sys
sys.path.append("/home/hieu.tranlm/sample_model/")
os.chdir("/home/hieu.tranlm/sample_model/")

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
```

### 🛠 Details
---
**1. Variable distribution**

There are some points that should be listed:
- Skewness: Many numerical variables exhibit right-skewness, which is common in financial data.
- Imbalance: The "BAD" variable shows a significant class imbalance, with far more "good" cases than "bad" cases.
- Categorical Dominance: The "REASON" variable is dominated by one category ("DebtCon").
- Missing Values: "Missing" categories are present in both REASON and JOB, indicating potential data quality issues.

![image](https://github.com/mynameishieu/Logistic-Regression-HMEQ/blob/main/docs/graphics/Variable%20distribution.png)

**2. Estimation of predictive power for variables**
- Most variavles have some degree of support for this classification, except for "REASON"
- The strongest variables are "DELINQ" and "DEBTINC", but "DEBTINC" should be checked for overpredicting as IV >= 0.5
- "CLAGE", "DEROG", "LOAN", "VALUE", and "JOB" have medium predictive power.
- The remaining variables, including "NINQ", "YOJ", "CLNO", and "MORTDUE", also have classification power, but they are weaker.

![image](https://github.com/mynameishieu/Logistic-Regression-HMEQ/blob/main/docs/graphics/Predictive%20power%20estimation%20-%20IV.png)

**3. Development Performance**

The performance recorded by the end of development cycle as follows:

***3.1. Accuracy***

The model has a high accuracy of over 85%, indicating that it performs well in correctly classifying the majority of cases
```
# Checking acccuracy on train/test sets
from sklearn.metrics import accuracy_score

y_pred_train = logit_model.predict(X_train)
acc_train = accuracy_score(y_pred_train, Y_train)
y_pred_test = logit_model.predict(X_test)
acc_test = accuracy_score(y_pred_test, Y_test)

print('accuracy on train: ', acc_train)
print('accuracy on test: ', acc_test)
```
```
accuracy on train:  0.8651426174496645
accuracy on test:  0.8557046979865772
```

***3.2. ROC Curve and AUC***

- The AUC (area under curve) index measures the area under the ROC curve, indicating whether the classification ability of the logistic regression model for GOOD/BAD cases is strong or weak.
- The larger AUC, the better the model.
- For this logistic regression model, AUC = 0.87 is quite high, indicating that the model's predictive ability is good and the model can be applied in practice.
 ![image](https://github.com/mynameishieu/Logistic-Regression-HMEQ/blob/main/docs/performance/ROC_curve.png)

***3.3. Precision and Recall***

**Overall Performance:** The curve indicates that the model has some discriminatory power. The precision and recall values are not consistently low, suggesting that the model can differentiate between positive and negative cases to some extent.

**Key Observations:**

- Trade-off: As expected, there's a clear trade-off between precision and recall. When precision is high, recall is low, and vice-versa. This is a fundamental characteristic of classification models.
- Intersection: The precision and recall curves intersect at a threshold around 0.4. This point represents a balance between precision and recall.
![image](https://github.com/mynameishieu/Logistic-Regression-HMEQ/blob/main/docs/performance/Precision_recall_curve.png)
- Precision is relative low at lower threshold, while the opposite is the case for the Recall value
```
precision on train:  0.501577287066246
precision on test:  0.4579831932773109
```
```
recall on train:  0.7383900928792569
recall on test:  0.7171052631578947
```

**4. Credit Score calculation**

The final step is to calculate the credit scorecard of each client by calculating the score for each feature which is considered as a bin range of a continuous variable or a class of a category variable. The score will be scaled according to the following formula:

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mtable displaystyle="true">
    <mlabeledtr>
      <mtd id="mjx-eqn:1">
        <mtext></mtext>
      </mtd>
      <mtd>
        <mtext>Score</mtext>
        <mo>=</mo>
        <mo stretchy="false">(</mo>
        <mi>&#x3B2;</mi>
        <mo>.</mo>
        <mtext>WOE</mtext>
        <mo>+</mo>
        <mfrac>
          <mi>&#x3B1;</mi>
          <mi>/n</mi>
        </mfrac>
        <mo stretchy="false">)</mo>
        <mo>.</mo>
        <mtext>Factor</mtext>
        <mo>+</mo>
        <mfrac>
          <mtext>Offset/</mtext>
          <mi>n</mi>
        </mfrac>
      </mtd>
    </mlabeledtr>
  </mtable>
</math>

---
In details:

- <math xmlns="http://www.w3.org/1998/Math/MathML">
  <mi>&#x3B2;: Coefficient of variable in logistic regression</mi>
</math> 

- <math xmlns="http://www.w3.org/1998/Math/MathML">
  <mi>&#x3B1;</math> : Intercept coefficient of logistic regression</mi>
- WOE: Weight of Evidence
- n: the number of variables
- Factor, Offset: parameters to calculate the Score, and calculated from pdo (Points to double the odds, pdo = 20) and Odds.
  + Factor = pdo/ln(2)
  + Offset = Base_Score - [Factor*ln(Odds)]

This model will choose to scale the points such that a base score of 600 points corresponds to good/bad odds of 50 to 1 and an increase of the score of 20 points corresponds to a doubling of the good/bad odds.

**Result: Score Distribution**

- Overall, the range of scores appears to be wide, indicating a significant spread in the data.
- There's a clear separation between the default and non-default distributions, indicating that the score effectively discriminates between the two groups.
![image](https://github.com/mynameishieu/Logistic-Regression-HMEQ/blob/main/docs/performance/Score_distribution.png)

**Conclusion**

A credit score evaluates a client's reliability in repaying debts to the bank. A higher credit score indicates greater trustworthiness, increasing the bank's confidence in the borrower. In addition, the bank can utilize credit scores to categorize customers into various risk or potential groups, similar to the FICO score system:

![image](https://www.investopedia.com/thmb/CMpgQ5ze8UECRpREBJhAg2xx1yc=/750x0/filters:no_upscale():max_bytes(150000):strip_icc():format(webp)/FICO-Scores-0474cc0ca87b4b58b9391f065f623c0f.jpg)

To mitigate risk, the bank may enforce stricter lending criteria or additional conditions. For instance, clients classified in the "Poor" category may be required to pay a loan fee, provide a deposit, or, in some cases, may be rejected a loan altogether.
