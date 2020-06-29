# Text Pre-processing
* We will remove any unnecessary qualities in the data which would make the trained model a poor generalizer.
* Text preprocessing involves things like removing emojis, properly formatting the text to remove extra spaces or any other information in the text that we don’t think would add value to our model.
* we need to ensure that the information we pass to the model is in a format that computers can understand. 
* Important to note that whatever we do on the train data should aso be done to the test data. 

### Pre-processing 1: Clean tweets text by removing links, special characters
* The function below uses regex to do bulk formatting for every tweet in the dataset.
* important to note that we can never have perfect data.

```python
    # helper function to clean tweets
    def processTweet(tweet):
        # Remove HTML special entities (e.g. &amp;)
        tweet = re.sub(r'\&\w*;', '', tweet)
        #Convert @username to AT_USER
        tweet = re.sub('@[^\s]+','',tweet)
        # Remove tickers
        tweet = re.sub(r'\$\w*', '', tweet)
        # To lowercase
        tweet = tweet.lower()
        # Remove hyperlinks
        tweet = re.sub(r'https?:\/\/.*\/\w*', '', tweet)
        # Remove hashtags
        tweet = re.sub(r'#\w*', '', tweet)
        # Remove Punctuation and split 's, 't, 've with a space for filter
        tweet = re.sub(r'[' + punctuation.replace('@', '') + ']+', ' ', tweet)
        # Remove words with 2 or fewer letters
        tweet = re.sub(r'\b\w{1,2}\b', '', tweet)
        # Remove whitespace (including new line characters)
        tweet = re.sub(r'\s\s+', ' ', tweet)
        # Remove single space remaining at the front of the tweet.
        tweet = tweet.lstrip(' ') 
        # Remove characters beyond Basic Multilingual Plane (BMP) of Unicode:
        tweet = ''.join(c for c in tweet if c <= '\uFFFF')
        return tweet
```

```python
    # clean dataframe's message column
    train_df['message'] = train_df['message'].apply(processTweet)
    # preview some cleaned tweets
    train_df.head(3)
```

|sentiment	| message | tweetid	| token_length | word count |
|---|---|---|---|---|
| 1	| polyscimajor epa chief doesn think carbon diox...	| 625221 | 19 | 19 |
| 1 | not like lack evidence anthropogenic global wa... |	126103 | 10 | 10 |
| 2 | researchers say have three years act climate c... | 698562 | 19 | 19 |


### Pre-processing 2: Tokenize without the Stop-Words
* As mentioned before, we do have some words in the dataset that are common in natural human language but used in most sentence compositions would be better left off since they bring no useful features to our model.
* we created our own stop words since some of the stop words contained in the pre loaded stopwords library are usefull for this particular analysis. for e.g "don't"
* also we have noted that removing all the stop words decreases the accuracy of our model.

```python
    # tokenize helper function
    def text_process(raw_text):
        """
        Takes in a string of text, then performs the following:
        1. Remove all punctuation
        2. Remove all stopwords
        3. Returns a list of the cleaned text
        """
        # Check characters to see if they are in punctuation
        nopunc = [char for char in list(raw_text) if char not in string.punctuation]
        # Join the characters again to form the string.
        nopunc = ''.join(nopunc)
        stopwords =['a','of'] 
        # Now just remove any stopwords
        return [word for word in nopunc.lower().split() if word.lower() not in stopwords]
```
* After removing stop-words we split all the sentences in the dataset to get individual words (tokens) which is basically a list of words per sentence contained in the newly processed tweet. Now we can see that we have two new columns in the dataframe that contains these tokenized versions of a tweet.

|sentiment	| message | tweetid	| token_length | word count | tokens |
|---|---|---|---|---|---|
| 1	| polyscimajor epa chief doesn think carbon diox...	| 625221 | 19 | 19 | [polyscimajor, epa, chief, doesn, think, carbo...|
| 1 | not like lack evidence anthropogenic global wa... |	126103 | 10 | 10 | [not, like, lack, evidence, anthropogenic, glo...|
| 2 | researchers say have three years act climate c... | 698562 | 19 | 19 | [researchers, say, have, three, years, act, cl...|

### Pre-processing 3: Feature Extraction
#### Vectorization — (Bag Of Words)
* We’ll convert each message which is represented by a list of tokens into a vector that a machine learning model can understand.
* We will use SciKit Learn’s CountVectorizer function which converts a collection of text documents to a matrix of token counts.
*  Since there are so many messages, we can expect a lot of zero counts for the presence of every word in the data but SciKit Learn will output a Sparse Matrix.
* we included an example of a vectorized text

```python
    # vectorize
    bow_transformer = CountVectorizer(analyzer=text_process).fit(train_df['message'])
    # print total number of vocab words
    print(len(bow_transformer.vocabulary_))

    # example of vectorized text
    sample_tweet = train_df['message'][125]
    print(sample_tweet)
    print('\n')
    # vector representation
    bow_sample = bow_transformer.transform([sample_tweet])
```
``` python
    # transform the entire DataFrame of messages
    messages_bow = bow_transformer.transform(train_df['message'])

    # check out the bag-of-words counts for the entire corpus as a large sparse matrix
    print('Shape of Sparse Matrix: ', messages_bow.shape)
    print('Amount of Non-Zero occurences: ', messages_bow.nnz)
```
```python
    Shape of Sparse Matrix:  (15819, 15747)
    Amount of Non-Zero occurences:  193457
```

# Term Frequency, Inverse Document Frequency
* TF-IDF stands for term frequency-inverse document frequency, and the tf-idf weight is a weight often used in information retrieval and text mining. This weight is a statistical measure used to evaluate how important a word is to a document in a collection or corpus.

To transform our entire twitter bag-of-words into TF-IDF collection at once we’ll use the code below:

```python
    # from sklearn.feature_extraction.text import TfidfTransformer
    tfidf_transformer = TfidfTransformer().fit(messages_bow)
    tfidf_sample = tfidf_transformer.transform(bow_sample)
    print(tfidf_sample)
```
```python
    (0, 15224)	0.24630811002138303
    (0, 14878)	0.2929559936510036
    (0, 14512)	0.4173447356453446
    (0, 11047)	0.2891471434212646
    (0, 9584)	0.36865763765820814
    (0, 2651)	0.08825623304746902
    (0, 2376)	0.08827344959271342
    (0, 2196)	0.39934776426974794
    (0, 2053)	0.28384162319345474
    (0, 305)	0.45187595592074714
```

```python
    # to transform the entire bag-of-words corpus
    messages_tfidf = tfidf_transformer.transform(messages_bow)
```

we are now ready to pass it through a ML classification algorithim.

# Model Training

### define X and y
* we will split the data into 80:20, this implies that 20% of our train data will be used for testing the model

```python
    # isolate label and text
    y = train_df['sentiment']
    X = train_df['message']
```

## Create Pipeline to manage the preprocessing steps in one step
* Scikit Learn library provides a pipeline capability that lets you define a pipeline workflow which will take all the above steps and even a classifier and grid search parameters.
* Pipelines make code more readable and also help avoid leaking statistics from your test data into the trained model in cross-validation, by ensuring that the same samples are used to train the transformers and predictors.
       nb: the process is a bit slow

### Cross Validation:
* The recommended method for training a good model is to first cross-validate using a portion of the training set itself to check if you have used a model whic is overfitting the data.
* To cross-validate and select the best parameter configuration at the same time, we use GridSearchCV.This allows us to easily test out different hyperparameter configuration.

```python
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV

    names = [
            "Naive Bayes",
            "Linear SVM",
            "Logistic Regression",
            "Random Forest",
            ]

    classifiers = [
        MultinomialNB(),
        LinearSVC(),
        LogisticRegression(solver='saga'),
        RandomForestClassifier(),
    ]

    parameters = [
                {'bow__ngram_range': [(1, 1), (1, 2)],
                    'tfidf__use_idf': (True, False),
                'clf__alpha': (1e-2, 1e-3)},
                {'bow__ngram_range': [(1, 1), (1, 2)],
                'tfidf__use_idf': (True, False),
                'clf__C': (np.logspace(-5, 1, 5))},
                {'bow__ngram_range': [(1, 1), (1, 2)],
                'tfidf__use_idf': (True, False),
                'clf__C': (np.logspace(-5, 1, 5))},
                {'bow__ngram_range': [(1, 1), (1, 2)],
                'tfidf__use_idf': (True, False),
                'clf__max_depth': (1, 2)},
                ]

    for name, classifier, params in zip(names, classifiers, parameters):
        clf_pipe = Pipeline([
            ('bow', CountVectorizer(strip_accents='ascii',
                                stop_words='english',
                                lowercase=True)),
            ('tfidf', TfidfTransformer()),
            ('clf', classifier),
        ])
        gs_clf = GridSearchCV(clf_pipe, param_grid=params, n_jobs=-1)
        clf = gs_clf.fit(X_train, y_train) 
        score = clf.score(X_test, y_test)
        #y_true, y_pred = y_test, gs_clf.predict(X_test)
        #print(classification_report(y_true, y_pred))
        print("{} score: {}".format(name, score))
```
Output:
```python
    Naive Bayes score: 0.7076485461441213
    Linear SVM score: 0.7316687737041719
    Logistic Regression score: 0.7319848293299621
    Random Forest score: 0.554677623261694
```
* After trying out the different model parameter combinations, the GridsearchCV returns the best performing model which we can use to classify new (twitter) data

## Logistic Regression
* below we singled out Logistic Regression model for further evaluation and for submissions purposes because it perfomed better(high accuracy) that the other models

```python
    # create pipeline
    pipeline = Pipeline([
        ('bow', CountVectorizer(strip_accents='ascii',
                                stop_words='english',
                                lowercase=True)),  # strings to token integer counts
        ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
        ('clf', LogisticRegression (solver = 'saga')),  # train on TF-IDF vectors w/ Naive Bayes classifier
    ])
    # this is where we define the values for GridSearchCV to iterate over
    parameters = {'bow__ngram_range': [(1, 1), (1, 2)],
                'tfidf__use_idf': (True, False),
                'clf__C': (np.logspace(-5, 1, 5)),
                }
    grid = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1)
    grid_fit = grid.fit(X_train,y_train)
    score = grid_fit.score(X_test, y_test)
    print(score)
    y_true, y_pred = y_test, grid.predict(X_test)
    print(classification_report(y_true, y_pred))
```
Output:
```python
    0.7319848293299621
              precision    recall  f1-score   support

          -1       0.74      0.41      0.53       278
           0       0.56      0.44      0.49       425
           1       0.76      0.86      0.81      1755
           2       0.74      0.71      0.72       706

   micro avg       0.73      0.73      0.73      3164
   macro avg       0.70      0.61      0.64      3164
weighted avg       0.73      0.73      0.72      3164
```
* In the cell above I used the best model to perform predictions on the unseen test data which lets us grade and retrieve the performance metrics.
* from the above we get perfomance metrics such as the classification report and a confusion matrix.