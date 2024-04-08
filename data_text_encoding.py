from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk

# Download the NLTK tokenizer models (only the first time)
nltk.download('punkt')


CORPUS = [
    "The quick brown fox jumps over the lazy dog",
    "A king's strength also includes his allies",
    "History is written by the victors",
    "An apple a day keeps the doctor away",
    "Nothing happens until something moves"
    ]
TARGET = 'apple'


if __name__ == "__main__":
    
    # BOW ENCODING
    bow_vectorizer = CountVectorizer()
    bow_matrix = bow_vectorizer.fit_transform(CORPUS)
    print(bow_vectorizer.get_feature_names_out())
    print(bow_matrix)

    # TF-IDF ENCODING
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(CORPUS)
    print(tfidf_vectorizer.get_feature_names_out())
    print(tfidf_matrix)


    # WORD2VEC ENCODING
    tokenized_docs = [word_tokenize(doc.lower()) for doc in CORPUS]
    word2vec_model = Word2Vec(tokenized_docs, vector_size=10, window=5, min_count=1, workers=4)
    
    # Save the model to disk
    # model.save("word2vec_example.model")
    # Load the model from disk
    # model = Word2Vec.load("word2vec_example.model")

    print(f"Encoding for {TARGET}:\n", word2vec_model.wv['apple'])
    similar_words = word2vec_model.wv.most_similar('apple', topn=5)
    print(f"Similar words to '{TARGET}':\n", similar_words)

