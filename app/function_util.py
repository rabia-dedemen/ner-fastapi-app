import spacy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import gensim
import nltk

nlp = spacy.load("en_core_web_sm")
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

    doc = nlp(text)
    filtered_tokens = [token.text for token in doc if token.ent_type_ not in ['PERSON', 'GPE']]

    tokens = gensim.utils.simple_preprocess(' '.join(filtered_tokens), deacc=True)
    return [wn.lemmatize(tok, simplify(pos)) for tok, pos in nltk.pos_tag(tokens) if tok not in stop_words]


def get_document_topics(model, bow):
    topics = model.get_document_topics(bow)
    return topics


def get_topic_name(topic_id):
    topic_names = ['Staff', 'Cleanliness', 'Location', 'Noise', 'Amenities', 'Check-in/out', 'Wi-Fi', 'Security', 'View']
    return topic_names[topic_id]
