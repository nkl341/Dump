from sklearn.feature_extraction.text import TfidfVectorizer

# Sample preprocessed conversations
preprocessed_conversations = [
    "you like pizza",
    "i prefer pasta over pizza",
    "pizza is my favorite food",
    "i don't like pizza"
]

# Create a TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit the vectorizer on the preprocessed conversations and transform the data
tfidf_matrix = tfidf_vectorizer.fit_transform(preprocessed_conversations)

# Convert the TF-IDF matrix to an array
tfidf_matrix_array = tfidf_matrix.toarray()

# Print the vocabulary (words/features)
print("Vocabulary:", tfidf_vectorizer.get_feature_names_out())

# Print the TF-IDF matrix
print("TF-IDF Matrix:")
print(tfidf_matrix_array)
