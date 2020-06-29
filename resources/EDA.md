# Explanatory Data Analysis

* We can observe that the data is indeed from tweet messages posted on twitter.
* The labels and the text do not seem to be in any listed order. This can be a problem if data is not randomly distributed as it can introduce biases to a learning model. So, we are going to use the Scikit Learn library which has a function to split our training and testing data and shuffle the data at the same time.
* We will also check the label frequency distribution in the data.
* We can also see that the text contains varying formats. Some words contain mixed case letters which need to be normalized to their base word.
* Leaving words with first letter capitalized can be experimented with as they may hold a different feature space like the name of a person or country etc.

```python
    train_df = pd.read_csv('resources/train.csv')
```

## Check for null values
```python
    >> train_df.info()
```
```
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 15819 entries, 0 to 15818
    Data columns (total 3 columns):
    sentiment    15819 non-null int64
    message      15819 non-null object
    tweetid      15819 non-null int64
    dtypes: int64(2), object(1)
    memory usage: 370.8+ KB
```

## Explore Class Distribution
```python
    >> train_df.sentiment.value_counts()
```
```
    1    8530
    2    3640
    0    2353
    -1    1296
    Name: sentiment, dtype: int64
```

## Sentiment Vocabulary

```python
    sentiment_explained = {'Sentiment_class': [2, 1, 0, -1],
                       'Type': ['News', 'Pro', 'Neutral', 'Anti'],
                       'Description': ["The tweet has a link to a real article on climate change",
                                       "The Tweet SUPPORTS the belief of man -made climate change",
                                       "The tweet does NOT support OR refute the belief of man-made climate change",
                                       "The Tweet does NOT believe in man-made climate change"],
                      'Sentiment_count': [3640, 8530, 2353, 1296]}

    sentiment_explained_df = pd.DataFrame(sentiment_explained)
    sentiment_explained_df
```

| Sentiment_class | Type | Description | Sentiment_count |
| --------------- | ---- | ----------- | --------------- |
| 2 | News | The tweet has a link to a real article on climate change| 3640 |
| 1 | Pro | The Tweet SUPPORTS the belief of man -made climate change |8530|
| 0	 | Neutral	| The tweet does NOT support OR refute the belief of man-made climate change |	2353 |
| -1 |	Anti | The tweet does NOT support OR refute the belief of man-made climate change | 1296 |

* The data is not evenly distributed, negative class has the least number of data entries with 1296, and the positive class has the most data with 8530 entries.We will neeed to rebalance the data so that can have a balanced dataset at least for training.
* This will be dealt with after the cleaning function is defined.


We needed to checked the number of words contained in every sentence so we created a function to extract this information and appended it to a column next to the text column. Below is a sample output:


```python
    # get a word count per sentence column
    def word_count(sentence):
        return len(sentence.split())
        
    train_df['word count'] = train_df['message'].apply(word_count)
    train_df.head(3)
```

| sentiment	| message | tweetid	| token_length| word count |
| --------- | ------- | ------- | ----------- | ---------- |
| 1| PolySciMajor EPA chief doesn't think carbon di... |	625221 |	19 |	19|
| 1 |	It's not like we lack evidence of anthropogeni... |	126103 | 	10 |	10 |
| 2 | RT @RawStory: Researchers say we have three ye... |	698562 | 19 | 19 |

![Sentiment Graph](/resources/sentiments.png)

## Common Words

We can observe
```python
    all_words = []
    for line in list(train_df['message']):
        words = line.split()
        for word in words:
            all_words.append(word.lower())
    
    Counter(all_words).most_common(10)
```
```python
    [('climate', 12323),
    ('rt', 9707),
    ('change', 8883),
    ('the', 7573),
    ('to', 7139),
    ('is', 4302),
    ('of', 4194),
    ('a', 4093),
    ('global', 3649),
    ('in', 3627)]
```

* In the cell above we extracted the most common words in the dataset and listed the top ten.
* We encountered words like 'is', 'the' and 'to' as they are very highly used in human expressions. These kind of words usually bring very little information that can be incorporated in the model so we will have to get rid of them down the road.
* Below is a code to output a graph showing the frequency of the first 25 words.

```python
    # plot word frequency distribution of first few words
    plt.figure(figsize=(12,5))
    plt.title('Top 25 most common words')
    plt.xticks(fontsize=13, rotation=90)
    fd = nltk.FreqDist(all_words)
    fd.plot(25,cumulative=False)
```
![Common Words](/resources/common.png)

## Word Cloud
* Below is  a word cloud showing the most common words in the entire twitter dataset after normalization.
* We can see our keywords climate & change are obviously very visible.Also, As expect a wide range of emotions are also very visible. Some words in the dataset do use very strong language.

```python
    # split sentences to get individual words
    all_words = []
    for line in train_df['tokens']: # try 'tokens'
        all_words.extend(line)

    # create a word frequency dictionary
    wordfreq = Counter(all_words)
    # draw a Word Cloud with word frequencies
    wordcloud = WordCloud(width=900,
                        height=500,
                        max_words=500,
                        max_font_size=100,
                        relative_scaling=0.5,
                        colormap='Blues',
                        normalize_plurals=True).generate_from_frequencies(wordfreq)
    plt.figure(figsize=(17,14))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()
```
![Word Cloud](/resources/wordcloud.png)
