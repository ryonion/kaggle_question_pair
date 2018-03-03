# kaggle_question_pair
determine whether two questions are duplicate or not with machine learning

- pre-processing:
  - stemming
  - abbreviation expanding

- top attributes:
  - Do the two questions share a starting word. (e.g. "what", "how")
  - Do the two questions share an ending symbol. (e.g. "?")
  - The number of shared word between the two questions.
  - sentence similarity (e.g. how similar is the nouns in the two questions)
  
- tools:
  - word2vec embedding
  - logistic regression
  - nltk
