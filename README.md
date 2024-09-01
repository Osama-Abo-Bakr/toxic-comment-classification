# Toxic Comment Classification

## Overview

This project focuses on classifying toxic comments using various Natural Language Processing (NLP) techniques. The aim is to preprocess the text data, extract meaningful features, build models, and evaluate their performance in accurately predicting different types of toxicity in comments.

## Project Structure

- **Data Reading:** Loading and inspecting the toxic comment dataset.
  
- **Data Preprocessing:** 
  - **Text Cleaning:** Converting text to lowercase, removing punctuation, and filtering out stop words.
  - **Lemmatization:** Reducing words to their base forms for consistency.
  
- **Feature Extraction:**
  - **TF-IDF Vectorization:** Converting text data into numerical features based on term frequency and inverse document frequency.
  - **Tokenization and Padding:** Breaking text into tokens and padding sequences to ensure uniform input size for the neural network.

- **Modeling:**
  - **Random Forest:** Implementing a Random Forest classifier to establish a baseline model using TF-IDF features and tokenized sequences.
  - **Deep Learning:** Building a Sequential Neural Network with embedding layers and training it to classify toxic comments.

- **Model Evaluation:**
  - **Accuracy Metrics:** Measuring the performance of models on training and testing datasets.
  - **Saving Models:** Saving the trained models and preprocessing tools for future use.

## Installation

To run this project, ensure you have the following dependencies installed:

```bash
pip install pandas numpy nltk scikit-learn tensorflow matplotlib pickle-mixin
```

## Usage

### Data Preprocessing

1. **Load the Dataset:**
   ```python
   data_train = pd.read_csv('train.csv')
   data_test = pd.read_csv('test.csv')
   ```

2. **Clean and Preprocess Text:**
   ```python
   def preprocess(data):
       # Add your text preprocessing steps here
       return processed_text

   data_train['text'] = data_train['comment_text'].apply(preprocess)
   data_test['text'] = data_test['comment_text'].apply(preprocess)
   ```

3. **Drop Unnecessary Columns:**
   ```python
   def drop(data):
       data = data.drop(columns=['id', 'comment_text'], axis=1)
       return data

   data_train = drop(data_train)
   data_test = drop(data_test)
   ```

### Feature Extraction

1. **TF-IDF Vectorization:**
   ```python
   tfidf = TfidfVectorizer()
   new_x = tfidf.fit_transform(data_train['text'])
   ```

2. **Tokenization and Padding:**
   ```python
   tokenize = Tokenizer()
   tokenize.fit_on_texts(data_train['text'])
   x_seq = tokenize.texts_to_sequences(data_train['text'])
   maxlen = max([len(seq) for seq in x_seq])
   x_pad = pad_sequences(x_seq, maxlen=maxlen, padding='pre')
   ```

### Model Building

1. **Train Random Forest Classifier:**
   ```python
   model_tfidf_rf = RandomForestClassifier()
   model_tfidf_rf.fit(x_train_TFIDF, y_train_TFIDF)
   ```

2. **Build and Train Neural Network:**
   ```python
   model = k.models.Sequential([
       Embedding(word_voc_length, 100, input_length=maxlen),
       GlobalAveragePooling1D(),
       Dense(128, activation="relu"),
       Dense(6, activation="softmax")
   ])
   model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
   model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
   ```

### Evaluation

Evaluate the model's performance using the training and testing datasets:

```python
print('Training Score (Random Forest TF-IDF): ', model_tfidf_rf.score(x_train_TFIDF, y_train_TFIDF))
print('Testing Score (Random Forest TF-IDF): ', model_tfidf_rf.score(x_test_TFIDF, y_test_TFIDF))

print('Training Score (Neural Network): ', model_tokenize.score(x_train, y_train))
print('Testing Score (Neural Network): ', model_tokenize.score(x_test, y_test))
```

### Saving Models

Save the trained models and vectorizers for future use:

```python
import pickle
pickle.dump(tfidf, open('tokenizer.bin', 'wb'))
pickle.dump(model_tfidf_rf, open('model.bin', 'wb'))
```

## Conclusion

This project demonstrates the application of machine learning and deep learning models in classifying toxic comments. By comparing traditional machine learning methods with deep learning approaches, the project offers insights into the effectiveness of different techniques in handling text classification tasks.

## Future Work

- **Hyperparameter Tuning:** Fine-tuning the models for improved accuracy.
- **Model Deployment:** Developing a web-based interface to classify comments in real-time.
- **Experimenting with other models:** Exploring different deep learning architectures like GRU or Transformer-based models.

---
