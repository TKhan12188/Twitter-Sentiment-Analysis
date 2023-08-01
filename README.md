# Twitter-Sentiment-Analysis

#### -- Project Status: [Completed]

## Project Intro/Objective
The purpose of this project was to build, train, and test an AI model for sentiment analysis on tweets from Twitter. The goal was to predict whether a tweet expressed a positive or negative sentiment.

### Methods Used
* Inferential Statistics
* Machine Learning
* Data Visualization
* Predictive Modeling
* etc.

### Technologies
* Python
* Pandas, jupyter
* etc.

## Project Description
* Dataset
  * The dataset used for this project consisted of 31,962 tweet samples with three main features: "tweet," "label" (indicating sentiment - 0 for positive, 1 for negative), and "id" (which was removed as it wasn't relevant for analysis).
    
* Data Preprocessing
  * Before analyzing the tweets, data preprocessing was performed using Python libraries such as NumPy, Pandas, and NLTK.
  * The preprocessing steps included:
    * Removing irrelevant columns (e.g., "id")
    * Removing punctuation and special characters from tweets
    * Removing stopwords (common words with little sentiment value)
    * Converting all the text to lowercase for uniformity
      
* Exploratory Data Analysis (EDA):
  * EDA was performed using Matplotlib and Seaborn to gain insights into the data distribution, sentiment balance, and word frequencies in positive and negative tweets. Word clouds were generated to visualize the most frequent words in each sentiment category.
  
* Count Vectorization:
  * Scikit-learn's CountVectorizer was used to convert the preprocessed text data into numerical feature vectors suitable for machine learning models. This process helped represent the textual data as numerical input for the Naive Bayes classifier.
  
* Model Training and Evaluation
  * A Naive Bayes classifier model was trained on the count vectors obtained from the tweets' text. The dataset was split into training and testing sets using the train_test_split function from Scikit-learn. Model performance was evaluated using classification metrics, including precision, recall, F1-score, and accuracy. The classification report and confusion matrix were used to assess the model's performance.
