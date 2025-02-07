# Sentiment Analysis using Machine Learning Models

## Overview
This project analyzes sentiment in Facebook posts using machine learning models, including:
- Support Vector Machine (SVM)
- Random Forest Classifier
- k-Nearest Neighbors (k-NN)

The dataset consists of 1,037 manually annotated Facebook posts, categorized into sentiment labels.

## Data Collection & Annotation
- Data was collected using a Facebook scraper.
- Posts were manually labeled as:
  - **Fact** (Neutral)
  - **Opinion** (Positive/Negative)

## Exploratory Data Analysis (EDA)
- Basic statistics and structure of the dataset
- Duplicate values removed
- Handling of missing values
- Data type conversion
- Class distribution visualization

## Data Preprocessing
- **Data Cleaning**: Removing special characters, stop words, Facebook-specific keywords, URLs, and user mentions.
- **Tokenization & Lemmatization**: Converting posts into word tokens and lemmatizing words.
- **Encoding Labels**: Mapping labels into numerical representations.
- **Data Splitting**: Dividing data into training and testing sets.
- **Feature Extraction**: Using TF-IDF vectorization.

## Model Training
### **1. Support Vector Machine (SVM)**
- Implemented with an RBF kernel and tuned hyperparameters.

### **2. Random Forest Classifier**
- Trained with 1000 estimators and a max depth of 12.

### **3. k-Nearest Neighbors (k-NN)**
- Used optimal `k=34` after cross-validation.

## Model Testing & Evaluation
- Performance metrics used:
  - **Accuracy**
  - **Precision**
  - **Recall**
  - **F1-score**
- Confusion matrices were plotted for each model.
- Precision-Recall Curves were analyzed.

## Results & Model Comparisons
| Model            | Accuracy | Precision | Recall | F1-score |
|-----------------|----------|----------|--------|---------|
| SVM             | 0.9064  | 0.9010   | 0.9381 | 0.9192  |
| Random Forest   | 0.8889  | 0.8679   | 0.9485 | 0.9064  |
| k-Nearest Neighbors | 0.8596  | 0.8925   | 0.8557 | 0.8737  |

## Future Work
- Enhance data augmentation for better model generalization.
- Fine-tune hyperparameters using grid search.
- Experiment with deep learning models like LSTMs or BERT for sentiment analysis.
- Expand dataset to cover more diverse opinions.

This markdown file serves as documentation for the sentiment analysis project, summarizing key insights and results.
