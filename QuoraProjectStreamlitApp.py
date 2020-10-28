import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
import pickle
import distance
import fuzzywuzzy as fuzz
from nltk.stem import PorterStemmer
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
import spacy #not
from xgboost import XGBClassifier
import nltk
import builtins
import warnings
import joblib
warnings.filterwarnings("ignore")
nltk.download('stopwords')
STOP_WORDS = stopwords.words("english")

st.title('Quora Question Pair Similarity')

#-----------------------------------------------------------------------CONSTANTS----------------------------------------------------------------------
#CONSTANTS
#Base url for all the files
# BASE_URL = "D:/AAIC/Module 6 - Machine Learning Real-World Case Studies/Case Study 1 - Quora question Pair Similarity Problem/StreamLit_Quora"
BASE_URL = "/home/ubuntu/quora"
#Data with basic features
DATA_URL_df_fe_without_preprocessing_train = BASE_URL+"/df_fe_without_preprocessing_train.csv"
#Data with advanced features
DATA_URL_nlp_features_train = BASE_URL+"/nlp_features_train.csv"
#Final features
DATA_URL_final_features = BASE_URL+"/final_feat_for_visualization.csv"

#SPACY file
NLP = spacy.load('en_core_web_sm')

SAFE_DIV = 0.0001 


#In final features dataframe there are 718 columns as they are trained on old Spacy modules
#New one gives 218 columns only.
COL_NAMES = ['cwc_min',
 'cwc_max',
 'csc_min',
 'csc_max',
 'ctc_min',
 'ctc_max',
 'last_word_eq',
 'first_word_eq',
 'abs_len_diff',
 'mean_len',
 'token_set_ratio',
 'token_sort_ratio',
 'fuzz_ratio',
 'fuzz_partial_ratio',
 'longest_substr_ratio',
 'freq_qid1',
 'freq_qid2',
 'q1len',
 'q2len',
 'q1_n_words',
 'q2_n_words',
 'word_Common',
 'word_Total',
 'word_share',
 'freq_q1+q2',
 'freq_q1-q2',
 '0_x',
 '1_x',
 '2_x',
 '3_x',
 '4_x',
 '5_x',
 '6_x',
 '7_x',
 '8_x',
 '9_x',
 '10_x',
 '11_x',
 '12_x',
 '13_x',
 '14_x',
 '15_x',
 '16_x',
 '17_x',
 '18_x',
 '19_x',
 '20_x',
 '21_x',
 '22_x',
 '23_x',
 '24_x',
 '25_x',
 '26_x',
 '27_x',
 '28_x',
 '29_x',
 '30_x',
 '31_x',
 '32_x',
 '33_x',
 '34_x',
 '35_x',
 '36_x',
 '37_x',
 '38_x',
 '39_x',
 '40_x',
 '41_x',
 '42_x',
 '43_x',
 '44_x',
 '45_x',
 '46_x',
 '47_x',
 '48_x',
 '49_x',
 '50_x',
 '51_x',
 '52_x',
 '53_x',
 '54_x',
 '55_x',
 '56_x',
 '57_x',
 '58_x',
 '59_x',
 '60_x',
 '61_x',
 '62_x',
 '63_x',
 '64_x',
 '65_x',
 '66_x',
 '67_x',
 '68_x',
 '69_x',
 '70_x',
 '71_x',
 '72_x',
 '73_x',
 '74_x',
 '75_x',
 '76_x',
 '77_x',
 '78_x',
 '79_x',
 '80_x',
 '81_x',
 '82_x',
 '83_x',
 '84_x',
 '85_x',
 '86_x',
 '87_x',
 '88_x',
 '89_x',
 '90_x',
 '91_x',
 '92_x',
 '93_x',
 '94_x',
 '95_x',
 '0_y',
 '1_y',
 '2_y',
 '3_y',
 '4_y',
 '5_y',
 '6_y',
 '7_y',
 '8_y',
 '9_y',
 '10_y',
 '11_y',
 '12_y',
 '13_y',
 '14_y',
 '15_y',
 '16_y',
 '17_y',
 '18_y',
 '19_y',
 '20_y',
 '21_y',
 '22_y',
 '23_y',
 '24_y',
 '25_y',
 '26_y',
 '27_y',
 '28_y',
 '29_y',
 '30_y',
 '31_y',
 '32_y',
 '33_y',
 '34_y',
 '35_y',
 '36_y',
 '37_y',
 '38_y',
 '39_y',
 '40_y',
 '41_y',
 '42_y',
 '43_y',
 '44_y',
 '45_y',
 '46_y',
 '47_y',
 '48_y',
 '49_y',
 '50_y',
 '51_y',
 '52_y',
 '53_y',
 '54_y',
 '55_y',
 '56_y',
 '57_y',
 '58_y',
 '59_y',
 '60_y',
 '61_y',
 '62_y',
 '63_y',
 '64_y',
 '65_y',
 '66_y',
 '67_y',
 '68_y',
 '69_y',
 '70_y',
 '71_y',
 '72_y',
 '73_y',
 '74_y',
 '75_y',
 '76_y',
 '77_y',
 '78_y',
 '79_y',
 '80_y',
 '81_y',
 '82_y',
 '83_y',
 '84_y',
 '85_y',
 '86_y',
 '87_y',
 '88_y',
 '89_y',
 '90_y',
 '91_y',
 '92_y',
 '93_y',
 '94_y',
 '95_y']
#----------------------------------------------------------------------------------------------------------------------------------------------------

#We need to host data somewhere!
@st.cache
def load_data(path_url,nrows):
	'''
		For visualization let's use some 10k rows only
	'''
	data = pd.read_csv(path_url,encoding='latin-1',nrows=nrows)
	return data

#Load the data, once laoded it will be stored in cache memory
data_load_state = st.text('Loading data...')
df = load_data(DATA_URL_df_fe_without_preprocessing_train,10000)
data_load_state = st.text('Loading data done !')

st.header('Basic extracted features')
basic_feature_list = '''<ol>
						<li><b>freq_qid1</b> = Frequency of qid1's</li>
						<li><b>freq_qid2</b> = Frequency of qid2's</li>
						<li><b>q1len</b> = Length of q1</li>
						<li><b>q2len</b> = Length of q2</li>
						<li><b>q1_n_words</b> = Number of words in Question 1</li>
						<li><b>q2_n_words</b> = Number of words in Question 2</li>
						<li><b>word_Common</b> = (Number of common unique words in Question 1 and Question 2)</li>
						<li><b>word_Total</b> =(Total num of words in Question 1 + Total num of words in Question 2)</li>
						<li><b>word_share</b> = (word_common)/(word_Total)</li>
						<li><b>freq_q1+freq_q2</b> = sum total of frequency of qid1 and qid2</li>
						<li><b>freq_q1-freq_q2</b> = absolute difference of frequency of qid1 and qid2</li>
						</ol>'''

st.markdown(basic_feature_list, unsafe_allow_html=True)

st.markdown('<hr style="height:4px;border-width:0;color:gray;background-color:blue">', unsafe_allow_html=True)

#Add checkbox to view data.
if st.checkbox('Show raw data'):
	st.subheader('Raw data')
	st.write(df)

st.markdown('<hr style="height:4px;border-width:0;color:gray;background-color:blue">', unsafe_allow_html=True)


#------------------------------------------------------------------------------------------------------------------------------
st.header("EDA on basic features")

#Word Share
st.subheader('Word Share')
fig = plt.figure(figsize=(7,3))
plt.subplot(1,2,1)
sns.violinplot(x = 'is_duplicate', y = 'word_share', data = df[0:])
plt.subplot(1,2,2)
sns.distplot(df[df['is_duplicate'] == 1.0]['word_share'][0:] , label = "1", color = 'red')
sns.distplot(df[df['is_duplicate'] == 0.0]['word_share'][0:] , label = "0" , color = 'blue' )
st.pyplot(fig, clear_figure=False)
st.markdown("<b>Questions which are duplicate have a higher number of word share count</b>", unsafe_allow_html=True)

#Duplicate vs Different
st.subheader('Similarity Count')
fig = plt.figure(figsize=(7,3))
# plt.subplots()
df.groupby("is_duplicate")['id'].count().plot.bar()
st.pyplot(fig, clear_figure=False)
st.markdown("<b>There are more questions which are not duplicate </b>", unsafe_allow_html=True)
st.markdown('<hr style="height:4px;border-width:0;color:gray;background-color:blue">', unsafe_allow_html=True)
#------------------------------------------------------------------------------------------------------------------------------

data_load_state = st.text('Loading data...')
df_adv_features = load_data(DATA_URL_nlp_features_train,10000)
data_load_state = st.text('Loading data done !')

st.header('Advanced extracted features')

st.markdown('<b>Definitions:-</b>', unsafe_allow_html=True)
definitions_string = '''<ol>
						<li><b>Token</b> = You get a token by splitting sentence a space</li>
						<li><b>Stop_Words</b> = stop words as per NLTK.</li>
						<li><b>Word</b> = A token that is not a stop_word</li></ol>'''
st.markdown(definitions_string, unsafe_allow_html=True)

st.markdown('<b>Features:-</b>', unsafe_allow_html=True)
advanced_feature_list = '''<ol>
						<li><b>cwc_min</b> = Ratio of common_word_count to min lenghth of word count of Q1 and Q2</li>
						<li><b>cwc_max</b> = Ratio of common_word_count to max lenghth of word count of Q1 and Q2</li>
						<li><b>csc_min</b> = Ratio of common_stop_count to min lenghth of stop count of Q1 and Q2</li>
						<li><b>csc_max</b> = Ratio of common_stop_count to max lenghth of stop count of Q1 and Q2</li>
						<li><b>ctc_min</b> = Ratio of common_token_count to min lenghth of token count of Q1 and Q2</li>
						<li><b>ctc_max</b> = Ratio of common_token_count to max lenghth of token count of Q1 and Q2</li>
						<li><b>last_word_eq</b> = Check if First word of both questions is equal or not</li>
						<li><b>first_word_eq</b> = Check if First word of both questions is equal or not</li>
						<li><b>abs_len_diff</b> = Abs. length difference</li>
						<li><b>mean_len</b> = Average Token Length of both Questions</li>
						<li><b>longest_substr_ratio</b> = Ratio of length longest common substring to min lenghth of token count of Q1 and Q2</li>
						</ol>'''
st.markdown(advanced_feature_list, unsafe_allow_html=True)

fuzzy_feature = '''<b>FuzzyWuzzy features:-</b><br>
				<a href="https://github.com/seatgeek/fuzzywuzzy#usage http://chairnerd.seatgeek.com/fuzzywuzzy-fuzzy-string-matching-in-python/">Read about FuzzyWuzzy here</a>
				'''
st.markdown(fuzzy_feature, unsafe_allow_html=True)
fuzzy_feature_list ='''<ol>
					<li><b>fuzz_ratio</b></li>
					<li><b>fuzz_partial_ratio</b></li>
					<li><b>token_sort_ratio</b></li>
					<li><b>token_set_ratio</b></li>
					</ol>'''
st.markdown(fuzzy_feature_list, unsafe_allow_html=True)

st.markdown('<hr style="height:4px;border-width:0;color:gray;background-color:blue">', unsafe_allow_html=True)

#Add checkbox to view data.
if st.checkbox('Show raw data for advance features'):
	st.subheader('Raw data advanced features')
	st.write(df_adv_features)

st.markdown('<hr style="height:4px;border-width:0;color:gray;background-color:blue">', unsafe_allow_html=True)
#------------------------------------------------------------------------------------------------------------------------------
st.header("EDA on advanced features")


st.subheader('Pair plot')
#No need to use plt.figure() here as sns.pairplot creates a figure of its own.
fig = sns.pairplot(df_adv_features[['ctc_min', 'cwc_min', 'csc_min', 'token_sort_ratio', 'is_duplicate']][0:1000], hue='is_duplicate',vars=['ctc_min', 'cwc_min', 'csc_min', 'token_sort_ratio'])
st.pyplot(fig, clear_figure=False)

st.subheader('Pair plot')
#No need to use plt.figure() here as sns.pairplot creates a figure of its own.
fig = sns.pairplot(df_adv_features[['fuzz_ratio', 'fuzz_partial_ratio', 'token_sort_ratio', 'longest_substr_ratio', 'is_duplicate']][0:1000], hue='is_duplicate',vars=['fuzz_ratio', 'fuzz_partial_ratio', 'token_sort_ratio', 'longest_substr_ratio'])
st.pyplot(fig, clear_figure=False)

#T-SNE visualization
st.subheader('T-SNE plot')

@st.cache
def prep_tsne_data(data):
	dfp_subsampled = data
	X = MinMaxScaler().fit_transform(dfp_subsampled[['cwc_min', 'cwc_max', 'csc_min', 'csc_max' , 'ctc_min' , 'ctc_max' , 'last_word_eq', 'first_word_eq' , 'abs_len_diff' , 'mean_len' , 'token_set_ratio' , 'token_sort_ratio' ,  'fuzz_ratio' , 'fuzz_partial_ratio' , 'longest_substr_ratio']])
	y = dfp_subsampled['is_duplicate'].values
	tsne_pickle_file = BASE_URL+"/tsne_data.pickle"
	# tsne2d = TSNE(
	#     n_components=2,
	#     init='random', # pca
	#     random_state=101,
	#     method='barnes_hut',
	#     n_iter=1000,
	#     verbose=2,
	#     angle=0.5
	# ).fit_transform(X)
	with open(tsne_pickle_file, 'rb') as handle:
		tsne2d = pickle.load(handle)
	return tsne2d,y

tsne2d,y = prep_tsne_data(df_adv_features[0:5000])
# with open('tsne_data.pickle', 'wb') as handle:
#     pickle.dump(tsne2d, handle, protocol=pickle.HIGHEST_PROTOCOL)

df_red = pd.DataFrame({'x':tsne2d[:,0], 'y':tsne2d[:,1] ,'label':y})

# draw the plot in appropriate place in the grid
fig = sns.lmplot(data=df_red, x='x', y='y', hue='label', fit_reg=False, size=8,palette="Set1",markers=['s','o'])
plt.title("perplexity : {} and max_iter : {}".format(30, 1000))
st.pyplot(fig, clear_figure=False)

st.markdown('<hr style="height:4px;border-width:0;color:gray;background-color:blue">', unsafe_allow_html=True)
#------------------------------------------------------------------------------------------------------------------------------
data_load_state = st.text('Loading data...')
final_features = load_data(DATA_URL_final_features,10000)
data_load_state = st.text('Loading data done !')

st.header('Final features')
st.text('Final features consists of: Advanced Features + Basic Features + TfIdfW2V_Ques1 + TfIdfW2V_Ques1 ')

#Add checkbox to view data.
# if st.checkbox('Show raw data for final features'):
# 	st.subheader('Raw data final features')
# 	st.write(final_features)

st.markdown('<hr style="height:4px;border-width:0;color:gray;background-color:blue">', unsafe_allow_html=True)
#------------------------------------------------------------------------------------------------------------------------------
st.header("EDA on final features")
#T-SNE visualization
st.subheader('T-SNE plot')

@st.cache
def prep_tsne_data(data):
	dfp_subsampled = data
	y = dfp_subsampled['is_duplicate'].values
	X = MinMaxScaler().fit_transform(dfp_subsampled.drop(['is_duplicate'], axis=1))
	tsne_pickle_file = BASE_URL+"/tsne_data_for_final_feat.pickle"
	# tsne2d = TSNE(
	#     n_components=2,
	#     init='random', # pca
	#     random_state=101,
	#     method='barnes_hut',
	#     n_iter=1000,
	#     verbose=2,
	#     angle=0.5
	# ).fit_transform(X)
	with open(tsne_pickle_file, 'rb') as handle:
		tsne2d = pickle.load(handle)
	return tsne2d,y

tsne2d,y = prep_tsne_data(final_features[0:5000])
# with open('tsne_data_for_final_feat.pickle', 'wb') as handle:
#     pickle.dump(tsne2d, handle, protocol=pickle.HIGHEST_PROTOCOL)

df_red = pd.DataFrame({'x':tsne2d[:,0], 'y':tsne2d[:,1] ,'label':y})

# draw the plot in appropriate place in the grid
fig = sns.lmplot(data=df_red, x='x', y='y', hue='label', fit_reg=False, size=8,palette="Set1",markers=['s','o'])
plt.title("perplexity : {} and max_iter : {}".format(30, 1000))
st.pyplot(fig, clear_figure=False)

st.markdown('<hr style="height:4px;border-width:0;color:gray;background-color:blue">', unsafe_allow_html=True)
#------------------------------------------------------------------------------------------------------------------------------
st.header("Modelling")
#------------------------------------------------------------------------------------------------------------------------------


import numpy as np
import pickle
import nltk
import distance
from fuzzywuzzy import fuzz
from nltk.stem import PorterStemmer
from bs4 import BeautifulSoup
import re
import spacy
from nltk.corpus import stopwords
import xgboost as xgb
nltk.download('stopwords')
import warnings
warnings.filterwarnings("ignore")
STOP_WORDS = stopwords.words("english")
SAFE_DIV = 0.0001 



#Featurize input query
def preprocess(x):
    x = str(x).lower()

    x = x.replace(",000,000", "m").replace(",000", "k").replace("′", "'").replace("’", "'").replace("won't", "will not").replace("cannot", "can not").replace("can't", "can not").replace("n't", " not").replace("what's", "what is").replace("it's", "it is").replace("'ve", " have").replace("i'm", "i am").replace("'re", " are").replace("he's", "he is").replace("she's", "she is").replace("'s", " own").replace("%", " percent ").replace("₹", " rupee ").replace("$", " dollar ").replace("€", " euro ").replace("'ll", " will")

    x = re.sub(r"([0-9]+)000000", r"\1m", x)
    x = re.sub(r"([0-9]+)000", r"\1k", x)


    porter = PorterStemmer()
    pattern = re.compile('\W')

    if type(x) == type(''):
        x = re.sub(pattern, ' ', x)


    if type(x) == type(''):
        x = porter.stem(x) #Perform stemming
        example1 = BeautifulSoup(x) 
        x = example1.get_text() #Get whole question text


    return x

def get_token_features(q1, q2):
    token_features = [0.0]*10

    # Converting the Sentence into Tokens: 
    q1_tokens = q1.split()
    q2_tokens = q2.split()

    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return token_features
    # Get the non-stopwords in Questions
    q1_words = set([word for word in q1_tokens if word not in STOP_WORDS])
    q2_words = set([word for word in q2_tokens if word not in STOP_WORDS])

    #Get the stopwords in Questions
    q1_stops = set([word for word in q1_tokens if word in STOP_WORDS])
    q2_stops = set([word for word in q2_tokens if word in STOP_WORDS])

    # Get the common non-stopwords from Question pair
    common_word_count = len(q1_words.intersection(q2_words))

    # Get the common stopwords from Question pair
    common_stop_count = len(q1_stops.intersection(q2_stops))

    # Get the common Tokens from Question pair
    common_token_count = len(set(q1_tokens).intersection(set(q2_tokens)))


    token_features[0] = common_word_count / (min(len(q1_words), len(q2_words)) + SAFE_DIV) #cwc_min
    token_features[1] = common_word_count / (max(len(q1_words), len(q2_words)) + SAFE_DIV) #cwc_max
    token_features[2] = common_stop_count / (min(len(q1_stops), len(q2_stops)) + SAFE_DIV) #csc_min
    token_features[3] = common_stop_count / (max(len(q1_stops), len(q2_stops)) + SAFE_DIV) #csc_max
    token_features[4] = common_token_count / (min(len(q1_tokens), len(q2_tokens)) + SAFE_DIV) #ctc_min
    token_features[5] = common_token_count / (max(len(q1_tokens), len(q2_tokens)) + SAFE_DIV) #ctc_max

    # Last word of both question is same or not
    token_features[6] = int(q1_tokens[-1] == q2_tokens[-1]) #last_word_eq

    # First word of both question is same or not
    token_features[7] = int(q1_tokens[0] == q2_tokens[0]) #first_word_eq

    token_features[8] = abs(len(q1_tokens) - len(q2_tokens)) #abs_len_diff

    #Average Token Length of both Questions
    token_features[9] = (len(q1_tokens) + len(q2_tokens))/2 #mean_len
    return token_features



def get_longest_substr_ratio(a, b):
    strs = list(distance.lcsubstrings(a, b))
    if len(strs) == 0:
        return 0
    else:
        return len(strs[0]) / (min(len(a), len(b)) + 1) #longest_substr_ratio

def extract_features(q1,q2):
    advanced_feature = []

    # preprocessing each question
    # Removing html tags,punctuations,stemming,stopwords,contractions, and then return the text of question
    q1 = preprocess(q1)
    q2 = preprocess(q2)


    token_features = get_token_features(q1,q2) #token_features is a list.
    advanced_feature.extend(token_features)
    #cwc_min,cwc_min,csc_min,csc_max,ctc_min,ctc_max,last_word_eq,first_word_eq,abs_len_diff,mean_len

    #fuzzy_features
    advanced_feature.append(fuzz.token_set_ratio(q1,q2))#token_set_ratio
    advanced_feature.append(fuzz.token_sort_ratio(q1,q2))#token_sort_ratio
    advanced_feature.append(fuzz.QRatio(q1,q2))#fuzz_ratio
    advanced_feature.append(fuzz.partial_ratio(q1,q2))#fuzz_partial_ratio
    advanced_feature.append(get_longest_substr_ratio(q1,q2))#longest_substr_ratio

    return advanced_feature



# print("len of dictionary --------------{}".format(len(word2tfidf)))
def get_w2v_feat(q,word2tfidf):
    doc1 = NLP(q)
    mean_vec1 = np.zeros([1, len(doc1[0].vector)])
    for word1 in doc1:
        vec1 = word1.vector
        try:
            idf = word2tfidf[str(word1)]
        except:
            idf = 0
        mean_vec1 += vec1 * idf
    # mean_vec1 = mean_vec1.mean(axis=0)
    return mean_vec1




def return_feature_vector(ques1, ques2):
    
    feat_vector = []
    
    #Advanced Features
    feat_vector=np.append(feat_vector,extract_features(ques1,ques2))

    #Basic Features
    feat_vector = np.append(feat_vector,[0])#freq_qid1
    feat_vector=np.append(feat_vector,[0]) #freq_qid2
    feat_vector=np.append(feat_vector,[len(ques1)]) #q1len
    feat_vector=np.append(feat_vector,[len(ques2)]) #q1len
    feat_vector=np.append(feat_vector,[(len(ques1.split(' ')))]) #q1_n_words
    feat_vector=np.append(feat_vector,[(len(ques2.split(' ')))]) #q2_n_words
    w1 = set(map(lambda word: word.lower().strip(), ques1.split(' ')))
    w2 = set(map(lambda word: word.lower().strip(), ques2.split(' ')))    
    feat_vector=np.append(feat_vector,[1.0 * len(w1 & w2)]) #word_Common
    feat_vector=np.append(feat_vector,[1.0 * (len(w1) + len(w2))]) #word_Total 
    feat_vector=np.append(feat_vector,[1.0 * len(w1 & w2)/(len(w1) + len(w2))]) #word_share
    feat_vector=np.append(feat_vector,[0])#freq_q1+q2
    feat_vector=np.append(feat_vector,[0])#freq_q1-q2

    word2tfidf_pickle_file = BASE_URL + "/tfidfnew.pickle"
    with open(word2tfidf_pickle_file, 'rb') as handle:
    	word2tfidf = pickle.load(handle)

    #w2vques1
    feat_vector=np.append(feat_vector,get_w2v_feat(ques1,word2tfidf))

    #w2vques2
    feat_vector=np.append(feat_vector,get_w2v_feat(ques2,word2tfidf))

    return feat_vector



test=return_feature_vector("Is abacus an intelligent exercise?", "Solving abacus requires intelligence?")

test = test.tolist()
lst = [test]

df = pd.DataFrame(lst, columns=COL_NAMES)
# print(df.values)


def get_model():
	xgboost_quora_model = BASE_URL+"/xgboost_xcfl_quora_model.model"
	x_cfl = XGBClassifier()
	x_cfl.load_model(xgboost_quora_model)
	return x_cfl
x_cfl = get_model()
pred = x_cfl.predict_proba(df)[0]
st.text(pred)
if (pred[0]>pred[1]):
	st.text("Questions are {} dissimilar".format(pred[0]*100))
else:
	st.text("Questions are {} similar".format(pred[1]*100))