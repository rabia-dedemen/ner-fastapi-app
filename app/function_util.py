from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import gensim

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def simplify(penn_tag):
    pre = penn_tag[0]
    if (pre == 'J'):
        return 'a'
    elif (pre == 'R'):
        return 'r'
    elif (pre == 'V'):
        return 'v'
    else:
        return 'n'

def preprocess(text):
    stop_words = stopwords.words('english')
    wn = WordNetLemmatizer()

    filtered_tokens = [word for word, pos in pos_tag(word_tokenize(text)) if pos not in ['PERSON', 'GPE']]

    tokens = gensim.utils.simple_preprocess(' '.join(filtered_tokens), deacc=True)
    return [wn.lemmatize(tok, simplify(pos)) for tok, pos in pos_tag(tokens) if tok not in stop_words]


def get_document_topics(model, bow):
    topics = model.get_document_topics(bow)
    return topics


def get_topic_name(topic_id):
    topic_names = ['Staff', 'Cleanliness', 'Location', 'Noise', 'Amenities', 'Check-in/out', 'Wi-Fi', 'Security', 'View']
    return topic_names[topic_id]
