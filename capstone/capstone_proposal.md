# Machine Learning Engineer Nanodegree
## Capstone Proposal
Stefan Langenbach 
March, 25th, 2019

## Proposal

### Domain Background
The challenge to forecast customer behaviour is prevalent in many industries. This project will investigate a specific 
variant of this problem in the financial services industry, specifically in consumer banking: To forecast which
customers will make a specific transaction in the future, irrespective of its amount.

For further information please consult the description of the corresponding _Kaggle Competiton_,
 [Santander Customer Transaction Prediction.](https://www.kaggle.com/c/santander-customer-transaction-prediction) 

### Problem Statement
The need for specific financial services and products is closely tied to customers' living situations, i.e. if they are
planning to attend college, start a family or buy real estate. In order to gain insight into their customers financial
situation, banks can use the history of their transactions. For the case at hand, the challenge is to **identify, which
customers will make a specific transaction in the future**, not taking into account the amount of that transaction. This
is a classical binary (0/1, yes/no) classification problem often found in the realm of data science.

### Datasets and Inputs
The data to be analyzed is provided by [Santander Bank](https://www.santanderbank.com/us/personal)
 through a Kaggle Competiton (see above). It is annonymized but its structure is identical to the data the bank uses 
 to solve similar problems.

For further information please consult the data section of the 
[Santander Customer Transaction Prediction competition.](https://www.kaggle.com/c/santander-customer-transaction-prediction/data)

### Solution Statement
A potential solution to the problem at hand can be obtained through the application of machine learning techniques.
By training various algorithms with historical transaction of individual customers, one can obtain models that can
predict the probability of a customer making a specific transaction.

### Benchmark Model
Given the [evaluation metric specified by the Kaggle Competition](https://www.kaggle.com/c/santander-customer-transaction-prediction#evaluation)
 (area under the ROC curve; see section below), the benchmark for the
solution is to better than random choice, i.e. reaching a ROC score > 0.5.
The personal ambition of the author is to come up with a solution placing him in the top 50% of the 
[Kaggle leaderboard](https://www.kaggle.com/c/santander-customer-transaction-prediction/leaderboard),
which translates into a ROC score of >= 0.89 (as of the time of writing this proposal).

### Evaluation Metrics
The evaluation metric used for this project is the 
[area under the ROC curve](https://en.wikipedia.org/wiki/Receiver_operating_characteristic) between the predicted
 probability of a customer making a specific transaction (calculated by a machine learning model) and the observed
 target, i.e. the actual transaction made by a customer in the past (available in the test dataset).

### Project Design
In order to arrive at a potential solution for this project, the following steps are required:

1. Set up a working, Python-based development environment
2. Gather relevant data from the corresponding Kaggle competition
3. Perform exploratory data analysis, data cleansing and feature engineering
4. Apply machine learning techniques (specifically implementations found in
 [fastai's](https://docs.fast.ai/) Python library) to build a model
5. Evaluate model performance by submitting model predictions to Kaggle and comparing them with the leaderboard
6. Write a report summarizing the project