import re
import nltk
import numpy
import pandas
from nltk.corpus import stopwords

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))
stemmer = nltk.stem.WordNetLemmatizer()
EMBEDDING_DIM=50
embeddings_path='models/embeddings/model.tsv'

def setup_nltk_resources():
    nltk.download('stopwords')
    nltk.download('wordnet')

def text_prepare(text):
    """
        text: a string

        return: modified initial string
    """
    text = text.lower()  # lowercase text
    text = re.sub(REPLACE_BY_SPACE_RE, ' ', text)  # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = re.sub(BAD_SYMBOLS_RE, ' ', text)  # delete symbols which are in BAD_SYMBOLS_RE from text
    text_words = text.split(' ')
    # lowercase,remove urls,delete stopwords and blanks
    resultwords = []
    for word in text_words:
        if len(word) <= 0: continue
        if word in STOPWORDS: continue
        if word.find('http') == 0: word = 'url'
        resultwords.append(word)

    # resultwords  = [word for word in text_words if word.lower() not in STOPWORDS and len(word)>0] # delete stopwords from text
    final_words = [stemmer.lemmatize(token) for token in resultwords]  # lemmatize words
    text = ' '.join(final_words)
    return text

def load_embeddings():
    dimheader = []
    for i in range(EMBEDDING_DIM):
        dimheader.append(str(i))
    embeddingsdf = pandas.read_table(embeddings_path, header=None, names=['word'] + dimheader)
    onlyembeddings = embeddingsdf.drop(['word'], axis=1).to_numpy()
    word2index = {}
    index2word = {}
    for i in range(embeddingsdf.shape[0]):
        word = embeddingsdf.iloc[i]['word']
        word2index[word] = i
        index2word[i] = word
    return onlyembeddings,word2index,index2word

def construct_user_vector(ut,onlyembeddings,word2index):
    utsents=ut.split('\t')
    utwords=[]
    for sent in utsents:
        words=sent.split(' ')
        utwords+=words
    utids=[]
    for word in utwords:
        try:
            wi=word2index[word]
        except:
            continue
        utids.append(wi)
    embeddingmatrix=onlyembeddings[[utids],:]
    embeddingmatrix=numpy.reshape(embeddingmatrix,(len(utids),onlyembeddings.shape[1]))
    user_embedding=numpy.mean(embeddingmatrix,axis=0)
    return(user_embedding)

def calculate_user_vector(sample_text):
    text_prepped=text_prepare(sample_text)
    oes,w2i,i2w=load_embeddings()
    tv=construct_user_vector(text_prepped,oes,w2i)
    return tv