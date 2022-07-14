# Machine Learning with Python
![Python](https://img.shields.io/badge/Python-3.7-brightgreen.svg)
![Machine Learning](https://img.shields.io/badge/AI-machine--learning-blue)
![Supervised-learning](https://img.shields.io/badge/ML-supervised--learning-orange)
![Unsupervised-learning](https://img.shields.io/badge/ML-unsupervised--learning-yellow)
![Reinforcement-learning](https://img.shields.io/badge/ML-reinforcement--learning-lightgrey)

## Overview

Machine Learning is a subset of *Artificial Intelligence* where Machine learning algorithms build models based on sample data (training data) and make decisions or predictions.

*"Machine learning (ML) is the study of computer algorithms that improve automatically through experience. Applications range from datamining programs that discover general rules in large data sets, to information filtering systems that automatically learn users' interests."* Mitchell, Tom 

There are 3 mains categories of Machine Learning that are divided based on "signal" or "feedback" used on the learning system:

### Supervised Machine Learning

Supervised learning uses an algorithm to learn the mapping function from an input to an output where both, input and output variables, are provided.

- **Groups:** regression (output variable as a real value) and classification (output variable as a category).

- **Popular Algorithms:** *Linear Regression*, *Logistic Regression*, *Random Forest* (classification and regression), *Decision Tree* (classification and regression), *Support Vector Machines* (classification) and *Naive Bayes* (classification).

- [Examples in this repo](./supervised-learning)


### Unsupervised Machine Learning

In unsupervised learning only input data is provided.

- **Groups:** *Clustering* (to discover the inherent groupings in the data) and *Association* (to identify rules that describe large sets of data).

- **Popular Algorithms:** *k-means* (clustering) and *Apriori* (association).

- [Examples in this repo](./unsupervised-learning)


### Reinforcement Machine Learning

In reinforcement learning the data is used to train an agent to learn to make decisions (take actions) in an interactive environment by trial and error using feedback from its own actions and experiences. 

- **Groups:** *Model-based RL* (uses experience to construct an internal model of the transitions and immediate outcomes in the environment) and *Model-free RL* (uses experience to learn directly one or both of two simpler quantities (state/ action values or policies) which can achieve the same optimal behavior but without estimation or use of a world model).

- **Popular Algorithms:** *Q-learning* and *SARSA* (State-Action-Reward-State-Action). 

- [Examples in this repo](./reinforcement-learning)



### This Repo

*In this repo I added multiple examples of machine learning algorithms:*

- `data` data sets used in the models
- `supervised-learning` sample notebooks with supervised-learning models
- `unsupervised-learning` sample notebooks with unsupervised-learning models
- `reinforcement-learning` sample notebooks with reinforcement-learning models


*This collection of Jupyter Notebooks came from my own studies and trainings I have done during the past years. They are all updated to be used with `Python 3.7`.*




### Machine Learning and Deep learning Notebooks

| Notebook Description| Link | Notes |
| -------------------| -----|--------|
| Iris Flower Classification | [Iris_flower_classification.ipynb](https://github.com/momer2020/Machine-Learning-Notebooks/blob/master/Iris_flower_classification.ipynb) | Build a neural network model using Keras & Tensorflow. Evaluated the model using scikit learn's k-fold cross validation. | 
| Recognizing CIFAR-10 images (Part I   - Simple model) | [Recognizing-CIFAR-10-images-Simple-Model.ipynb](https://github.com/momer2020/Machine-Learning-Notebooks/blob/master/Recognizing-CIFAR-10-images-Simple-Model.ipynb) | Build a simple Convolutional Neural Network(CNN) model to classify CIFAR-10 image dataset with Keras deep learning library achieving classification accuracy of 67.1%. |
| Recognizing CIFAR-10 images (Part II  - Improved model) | [Recognizing-CIFAR-10-images-Simple-Model.ipynb](https://github.com/momer2020/Machine-Learning-Notebooks/blob/master/Recognizing-CIFAR-10-images-Improved-Model.ipynb) | Build an improved CNN model by adding more layers with Keras deep learning library achieving classification accuracy of 78.65%. |
| Recognizing CIFAR-10 images (Part III - Data Augmentation) | [Recognizing-CIFAR-10-images-Improved-Model-Data-Augmentation.ipynb](https://github.com/momer2020/Machine-Learning-Notebooks/blob/master/Recognizing-CIFAR-10-images-Improved-Model-Data-Augmentation.ipynb) | Build an improved CNN model by data augmentation with Keras deep learning library achieving classification accuracy of 80.73%. |
| Traffic Sign Recognition using Deep Learning | [Traffic-Sign-Recognition.ipynb](https://github.com/momer2020/Machine-Learning-Notebooks/blob/master/Traffic-Sign-Recognition.ipynb) | Build a deep learning model to detect traffic signs using the German Traffic Sign Recognition Benchmark(GTSRB) dataset achieving an accuracy of 98.4%. |
| Movie Recommendation Engine | [Movie_Recommendation_Engine.ipynb](https://github.com/momer2020/Machine-Learning-Notebooks/blob/master/Movie_Recommendation_Engine.ipynb) | Build a movie recommendation engine using k-nearest neighbour algorithm implemented from scratch. |
| Linear Regression | [Linear_Regression.ipynb](https://github.com/momer2020/Machine-Learning-Notebooks/blob/master/Linear_Regression.ipynb) | Build a simple linear regression model to predict profit of food truck based on population and profit of different cities. |
| Multivariate Linear Regression | [Multivariate_Linear_Regression.ipynb](https://github.com/momer2020/Machine-Learning-Notebooks/blob/master/Multivariate_Linear_Regression.ipynb) | Build a simple multivariate linear regression model to predict the price of a house based on the size of the house in square feet and number of bedrooms in the house. |
| Sentiment Analysis of Movie Reviews| [Sentiment_Analysis.ipynb](https://github.com/momer2020/Machine-Learning-Notebooks/blob/master/Sentiment_Analysis.ipynb)| Experiment to analyze sentiment according to their movie reviews. |
| Wine quality prediction | [Predicting_wine_quality.ipynb](https://github.com/momer2020/Machine-Learning-Notebooks/blob/master/Predicting_wine_quality.ipynb)| Experiment to predict wine quality with feature selection (In progress). |
| Unsupervised Learning | [unsupervised_learning-Part_1.ipynb](https://github.com/momer2020/Machine-Learning-Notebooks/blob/master/unsupervised_learning-Part_1.ipynb)| Hands-on with Unsupervised learning. |
| Autoencoders using Fashion MNIST| [Autoencoder_Fashion_MNIST.ipynb](https://github.com/momer2020/Machine-Learning-Notebooks/blob/master/Autoencoder_Fashion_MNIST.ipynb)| Building an autoencoder as a classifier using Fashion MNIST dataset. |
| Logistic Regression| [Logistic_Regression.ipynb](https://github.com/momer2020/Machine-Learning-Notebooks/blob/master/Logistic_Regression.ipynb)| Build a logistic regression model from scratch - Redoing it |
| Fuzzy string matching| [fuzzywuzzy.ipynb](https://github.com/momer2020/Machine-Learning-Notebooks/blob/master/fuzzy_string_matching.ipynb)| To study how to compare strings and determine how similar they are in Python. |
| Spam email classification| [spam_email_classification.ipynb](https://github.com/momer2020/Machine-Learning-Notebooks/blob/master/spam_email_classification.ipynb)| Build a spam detection classification model using an email dataset.
| Customer churn prediction| [customer_churn_prediction.ipynb](https://github.com/momer2020/Machine-Learning-Notebooks/blob/master/customer_churn_prediction.ipynb)| To predict if customers churn i.e. unsubscribed or cancelled their service.- In Progress|
| Predicting Credit Card Approvals| [predicting_credit_card_approvals.ipynb](https://github.com/momer2020/Machine-Learning-Notebooks/blob/master/predicting_credit_card_approvals.ipynb)| To predict the approval or rejection of a credit card application|
