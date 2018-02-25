import os
import pandas as pd
import string
import nltk
import nltk.corpus
from nltk.corpus import stopwords
import gensim
from gensim import corpora, models
from time import gmtime, strftime
from collections import Counter
import itertools
import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pickle



# ORGANIZES THREE SEPERATE CORPUS FOR INPUT DATA
def cmu_data_matrix(texts, tsv):

    
    with open(texts, encoding='utf-8') as text_file:
        lines = text_file.readlines()
        texto = [i.split('\t') for i in lines]

    # resolve refrences 
        text_dict = {int(iD):texts for iD, texts in texto }

    # initiate dataframe
    column_names = ['Wiki movie ID', 'Freebase movie ID', 'Movie Name', 'Movie release data', 'Movie box office', 'Movie runtime',
          'Movie languages', 'Movie Countries', 'Movie Genres']
    
    #populate database
    movie_data_matrix = pd.read_csv(tsv, sep='\t', names=column_names)
    movie_data_matrix['text'] = movie_data_matrix['Wiki movie ID'].map(text_dict)
    movie_data_matrix['genre_data'] =[set(eval(i.lower()).values()) for i in movie_data_matrix['Movie Genres']]
    
    print('Intial shape of data %s' %str(movie_data_matrix.shape))
    
    #filter by language and available texts
    movie_data_matrix_english=movie_data_matrix.loc[movie_data_matrix['Movie languages'].str.contains("english", case=False)]
    print('Shape of data for english films %s' %str(movie_data_matrix_english.shape))
    matrix = movie_data_matrix_english.dropna(subset=['text'])
    print('Shape of data for english films with source text %s' %str(matrix.shape))
    
    corpus = matrix.text.tolist()
    print('Final size of corpus %s' %str(len(corpus)))
    
    #create genre_labels 
    genre_distribution = Counter(itertools.chain(*matrix['genre_data']))
    del genre_distribution['drama']
    del genre_distribution['comedy'] 
    
    matrix = matrix.dropna(subset=['genre_data'])
   
    
    matrix['labels'] = [sorted(list((genre_distribution[x],x) for x in genre), reverse=True)[:1]
                        for genre in matrix['genre_data']]
    print('Shape of data for english films with source text and genre labels %s' %str(matrix.shape))

    matrix.to_csv('cmu_data.csv')
    
    return matrix

def film2_data_matrix():

    path = "/Users/ViVeri/Desktop/ISO_Networks/imsdb_raw_nov_2015"

    name = []
    filenames = []
    text = []

    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".txt"):
                

                name.append(file)
                filenames.append(os.path.join(root, file))

    for i in filenames:
        with open(i) as inputfile:
            screenplay_raw = inputfile.read().strip('\t\n\r')
            screenplay=' '.join(screenplay_raw.split())
            text.append(screenplay)
            
            


    labels =[os.path.dirname(i).split('/')[-1] for i in filenames]
    
    matrix =pd.DataFrame({'name':name, 'text': text, 'labels':labels})

    matrix.to_csv('film2_data.csv')

    print('Shape of data for Film2.0 corpus %s' %str(matrix.shape))

    return matrix
    
def tv_data_matrix():

    path = "/Users/ViVeri/Desktop/ISO_Networks/tv_raw_2018"

    name = []
    filenames = []
    text = []

    #access all files in tv corpus directory 
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".txt"):
        
                name.append(file)
                filenames.append(os.path.join(root, file))

    #read all of those files and append to list
    for i in filenames:
        with open(i) as inputfile:
            text.append(inputfile.read().replace('\t',"" ).replace('\n', '').strip())
            


    show =[os.path.dirname(i).split('/')[-2] for i in filenames]
    season = [os.path.dirname(i).split('/')[-1] for i in filenames]
    episode_num=[i.split('.')[0] for i in name]

    
    matrix =pd.DataFrame({'title_episode':name, 'text': text, 
                          'name':show, 'season':season, 'episode_num':episode_num})

    print('Shape of data for TV_corpus %s' %str(matrix.shape))

    matrix.to_csv('tv_data.csv')
    
    return matrix

def universe_of_discourse():
		C = cmu_data_matrix('plot_summaries.txt','movie.metadata.tsv')
		print('CMU collected')
		F = film2_data_matrix()
		print('Film2.0 collected')
		T = tv_data_matrix()
		print('TV_2018 collected')

		return C, F, T

# RUN TO CREATE INTIAL FILES 
# x = universe_of_discourse()

#RUN WHEN FILES ARE CREATED
def load_universe():
	C = pd.DataFrame.from_csv('cmu_data.csv')
	F = pd.DataFrame.from_csv('film2_data.csv')
	T = pd.DataFrame.from_csv('tv_data.csv')

	return C, F , T

# DELIVERS LIST OF CURATED STOP WORDS 



# Machine Learning MODEL 
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn import linear_model, decomposition
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression

def stop_words():
    #common stop words
    stop_words1 = stopwords.words('english')
    
    #common name stop words
    with open('name_stopwords.tsv') as text_file:
        stop_words2 = text_file.read().lower().replace(',', " ").split()

    with open('additional_stop.txt', 'r') as stop:
    	stop_words_add = stop.read().split()
    
    stop_words3 = [i.lower() for i in nltk.corpus.names.words()]
    
    #common punctuation and letter
    punctuation = [i for i in string.punctuation] + ["'s", '...', '-PRON-', 'int', 'ext',' ', '\n','--']
    
    letters = list(string.ascii_lowercase)
    
    stopwords_theta= set(stop_words1+stop_words2+stop_words3 + punctuation+ letters+stop_words_add)
    
    return stopwords_theta


# pipeline = Pipeline([
#     ('tfidf', TfidfVectorizer(stop_words=stop_words()),
#     ('clf', OneVsRestClassifier(LogisticRegression(solver='sag'))))])

# parameters = {
#     'tfidf__max_df': (0.25, 0.5, 0.75),
#     'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
#     "clf__estimator__C": [0.01, 0.1, 1],
#     "clf__estimator__class_weight": ['balanced', None],
# }


def train_test_split(data_matrix):
	data_x = data_matrix[['text']].as_matrix()
	data_y = data_matrix.drop(['name', 'text'], axis=1).as_matrix()

	stratified_split = StratifiedShuffleSplit(n_splits=2, test_size=0.33)
	for train_index, test_index in stratified_split.split(data_x, data_y):
	    x_train, x_test = data_x[train_index], data_x[test_index]
	    y_train, y_test = data_y[train_index], data_y[test_index]

	# transform matrix of plots into lists to pass to a TfidfVectorizer
	train_x = [x[0].strip() for x in x_train.tolist()]
	train_y = [x[0].strip() for x in y_train.tolist()]
	test_x = [x[0].strip() for x in x_test.tolist()]
	test_y = [x[0].strip() for x in y_test.tolist()]

	return train_x, train_y, test_x, test_y





# grid_search_tune = GridSearchCV(
#     pipeline, parameters, cv=2, n_jobs=2, verbose=3)
# grid_search_tune.fit(train_x, train_y)


# print("Best parameters set:")
# print (grid_search_tune.best_estimator_.steps)


# # measuring performance on test set
# print ("Applying best classifier on test data:")
# best_clf = grid_search_tune.best_estimator_
# predictions = best_clf.predict(test_x)

# print (classification_report(test_y, predictions, target_names=labels))







