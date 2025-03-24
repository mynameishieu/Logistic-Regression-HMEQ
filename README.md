# Logistic Regression model - Home Equity dataset (HMEQ)

### 📌 Project Overview
---
This is a project from Kaggle to explore a dataset concerning the loan status on a home equity line of credit. (https://www.kaggle.com/datasets/ajay1735/hmeq-data)

**Context:** The consumer credit department of a bank wants to automate the decisionmaking process for approval of home equity lines of credit. To do this, they will follow the recommendations of the Equal Credit Opportunity Act to create an empirically derived and statistically sound credit scoring model. The model will be based on data collected from recent applicants granted credit through the current process of loan underwriting. The model will be built from predictive modeling tools, but the created model must be sufficiently interpretable to provide a reason for any adverse actions (rejections).
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
