# Kindle Book Rating Classifier

Can we predict whether a Kindle book will get a 5-star rating? This notebook combines a book's numerical data (price, number of pages, file size, number of customer reviews), its device features (text-to-speech, X-Ray, lending) and the text of its description.

## Approach

1. Explore distributions, missing values and correlations, and build the target: 5 stars vs. fewer than 5.
2. Train a baseline with the numerical and device features.
3. Add the description text with TF-IDF and LSA (truncated SVD), choosing the number of topics by explained variance.
4. Log-transform the skewed numerical features and retrain.

Models compared: logistic regression, Gaussian Naive Bayes, decision tree and random forest.

## Results (test set, final run)

| Model | ROC AUC |
|---|---|
| Logistic regression | 0.93 |
| Random forest | 0.92 |
| Gaussian Naive Bayes | 0.88 |
| Decision tree | 0.65 |

## Data

Dataset available at [BigML](https://bigml.com/user/ashikiar/gallery/dataset/53f3ba12547dfc64bd000a1f). Download it and update the file path at the top of the notebook.

## Stack

Python, pandas, NumPy, SciPy, scikit-learn, Matplotlib, Seaborn.
