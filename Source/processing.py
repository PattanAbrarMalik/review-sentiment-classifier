import nltk

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("wordnet")
nltk.download("omw-1.4")

from nltk.corpus import stopwords

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, LancasterStemmer
from nltk.tokenize import word_tokenize, RegexpTokenizer


sw = stopwords.words('english')
# Keep negation words — removing them flips sentiment meaning
# (e.g. "not fantastic" becoming just "fantastic")
negation_words = {
    "not", "no", "nor", "never", "none", "nothing", "nobody", "neither",
    "n't", "cannot", "isn't", "aren't", "wasn't", "weren't", "don't",
    "doesn't", "didn't", "hasn't", "haven't", "hadn't", "won't", "wouldn't",
    "shan't", "shouldn't", "can't", "couldn't", "mustn't"
}
sw = [w for w in sw if w not in negation_words]
tokenizer = RegexpTokenizer(r'\w+')


def process_text(review, stem='p'):
    """
    Given a text, the function converts the text into lower case,
    removes stopwords, removes punctuations, tokenizes the text,
    performs stemming and returns the processed text
    :param review: raw text
    :param stem: Stemmer - 'p' for PorterStemmer and 'l' for
                        LancasterStemmer
    :return: processed text
    """
    # Convert text to lower
    review = review.lower()
    # Word tokenize the review
    tokens = word_tokenize(review)
    # Remove stopwords
    tokens = [t for t in tokens if t not in sw]
    # Remove punctuation
    tokens = [tokenizer.tokenize(t) for t in tokens]
    tokens = [t for t in tokens if len(t)>0]
    tokens = ["".join(t) for t in tokens]
    # Create stemmer
    if stem == 'p':
        stemmer = PorterStemmer()
    elif stem == 'l':
        stemmer = LancasterStemmer()
    else:
        raise Exception("stem has to be either 'p' for Porter or 'l' for Lancaster")
    # Stemming
    tokens = [stemmer.stem(t) for t in tokens]
    # Return clean string
    return " ".join(tokens)
