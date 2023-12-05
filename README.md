# Polynimialai
# Internship Recruitment Drive 2023

## Machine Learning / NLP
### Introduction of the problem
The problem statement goes as follows:

Given a dataset of Amazon product reviews for the year 2017-2018 for the category [Cell Phones and accessories](http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/Cell_Phones_and_Accessories.json.gz) along with some [metadata](http://deepyeti.ucsd.edu/jianmo/amazon/metaFiles2/meta_Cell_Phones_and_Accessories.json.gz), you are required to do the following:


**Note**: If someone faces issue while downloading the dataset , please go to http://deepyeti.ucsd.edu/jianmo/amazon/ and download the *Cell_Phones_and_Accessories.json.gz* directly from here along with its meta data


1. Reading and pre-processing of the dataset (Hint: As the dataset is huge, the normal way of loading the dataset might cause some memory issues):
2. Creating a classifier for the classification of Reviews into positive, negative, and neutral: 
3. Create a Confusion matrix and support training and Testing metrics: 


# Sentiment Analysis on Cell Phone Reviews

## Approach Used

### Data Loading and Exploration

- Loaded the dataset using Google Colab and Pandas.
- Checked for null values in the dataset.
- Calculated the average rating in the dataset.

### Text Preprocessing with NLTK

- Tokenized the text using the NLTK library.
- Lowercased all text for consistency.
- Applied stemming using Porter Stemmer.
- Removed stop words for meaningful analysis.

### Model Creation using Multinomial Naive Bayes

- Used the scikit-learn library for feature extraction and model creation.
- Utilized the CountVectorizer for converting text data into a bag-of-words format.
- Split the dataset into training and testing sets.
- Applied Multinomial Naive Bayes as the classification algorithm.

### Model Evaluation

- Created a confusion matrix to evaluate the performance of the model.
- Calculated and printed the predicted probabilities for each class.
- Determined the overall accuracy of the model on the test set.

## Model Architecture

- **CountVectorizer:** Converted the processed text data into numerical features.
- **Multinomial Naive Bayes:** Chosen as the classification algorithm for its simplicity and effectiveness in text classification tasks.

## Model Accuracy

- The model achieved an accuracy of `53%` on the test set.


# Sentiment Analysis on Cell Phone Reviews using SVM

## Approach Used

### Data Loading and Exploration

- Loaded the dataset using Google Colab and Pandas.
- Checked for null values in the dataset.
- Calculated the average rating in the dataset.

### Text Preprocessing with NLTK

- Tokenized the text using the NLTK library.
- Lowercased all text for consistency.
- Applied stemming using Porter Stemmer.
- Removed stop words for meaningful analysis.

### Model Creation using Support Vector Machine (SVM)

- Used the scikit-learn library for feature extraction and model creation.
- Utilized the CountVectorizer for converting text data into a bag-of-words format.
- Employed SVM with a linear kernel for classification.
- Enabled probability estimates for more informative analysis.

### Model Evaluation

- Created a confusion matrix to evaluate the performance of the SVM model.
- Calculated and printed the predicted probabilities for each class.
- Determined the overall accuracy of the SVM model on the test set.

## Model Architecture

- **CountVectorizer:** Converted the processed text data into numerical features.
- **Support Vector Machine (SVM):** Chosen as the classification algorithm with a linear kernel for its effectiveness in text classification tasks.

## Model Accuracy and Classification Report

- The SVM model achieved an accuracy of `51%` on the test set.
- The classification report provides a detailed breakdown of precision, recall, and F1-score for each class.

## Confusion Matrix

- Visualized the confusion matrix to understand the performance of the SVM model.



