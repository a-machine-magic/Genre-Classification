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




def data_wrangler(texts,tsv):
    
    #access data files 
    with open(texts, encoding='utf-8') as text_file:
        lines = text_file.readlines()
        text = [i.split('\t') for i in lines]

    # resolve refrences 
        text_dict = {int(iD):texts for iD, texts in text }

    # initiate dataframe
    column_names = ['Wiki movie ID', 'Freebase movie ID', 'Movie Name', 'Movie release data', 'Movie box office', 'Movie runtime',
          'Movie languages', 'Movie Countries', 'Movie Genres']
    
    #populate database
    movie_data_matrix = pd.read_csv(tsv, sep='\t', names=column_names)
    movie_data_matrix['Text'] = movie_data_matrix['Wiki movie ID'].map(text_dict)
    movie_data_matrix['genre_data'] =[set(eval(i.lower()).values()) for i in movie_data_matrix['Movie Genres']]
    
    print('Intial shape of data %s' %str(movie_data_matrix.shape))
    
    #filter by language and available texts
    movie_data_matrix_english=movie_data_matrix.loc[movie_data_matrix['Movie languages'].str.contains("english", case=False)]
    print('Shape of data for english films %s' %str(movie_data_matrix_english.shape))
    matrix = movie_data_matrix_english.dropna(subset=['Text'])
    print('Shape of data for english films with source text %s' %str(matrix.shape))
    
    corpus = matrix.Text.tolist()
    print('Final size of corpus %s' %str(len(corpus)))
    
    #create genre_labels 
    genre_distribution = Counter(itertools.chain(*matrix['genre_data']))
    del genre_distribution['drama']
    del genre_distribution['comedy'] 
    
    matrix = matrix.dropna(subset=['genre_data'])
   
    
    matrix['labels'] = [sorted(list((genre_distribution[x],x) for x in genre), reverse=True)[:1]
                        for genre in matrix['genre_data']]
    print('Shape of data for english films with source text and genre labels %s' %str(matrix.shape))
    
    return movie_data_matrix,matrix,corpus


def preprocessed_data(corpus):
    
    stop_words1 = stopwords.words('english')
    
    with open('name_stopwords.tsv') as text_file:
        stop_words2 = text_file.read().lower().replace(',', " ").split()
    
    stop_words3 = [i.lower() for i in nltk.corpus.names.words()]
    
    stopwords_theta= set(stop_words1+stop_words2+stop_words3)
    ########################################################################
    punctuation = list(string.punctuation)
    keep_punctuation = ["'", "-"]
    
    global punctuation_beta
    punctuation_beta=set(punctuation) - set(keep_punctuation)
    ########################################################################
    
    tokenize = lambda x:x.lower().replace('.', " ").replace(',', " ").replace('"', " ").split()
    tokenized_corpus = [list(tokenize(i)) for i in corpus]
    
    cleaned_documents=[list(token for token in document 
                           if not any(p in token for p in punctuation_beta) and
                               token not in (stopwords_theta or punctuation_beta)) 
                                   for document in tokenized_corpus]


    pickle.dump(cleaned_documents, open( "cleaned_documents.p", "wb" ) )
    
    return cleaned_documents

def LDA_Model(tokenized_documents, topics, passes_value):
    dictionary=corpora.Dictionary(tokenized_documents)
    corpus = [dictionary.doc2bow(text) for text in tokenized_documents]
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=topics, id2word = dictionary, passes=passes_value)
    return ldamodel, corpus, dictionary

def topic_items(LDA_model, num_words):
    legend ={}
    for i in  LDA_model.show_topics(num_topics=LDA_model.num_topics,num_words=num_words, formatted=False):
        key = [j for j,_ in i[1]]
#         print (i[0], [j for j,_ in i[1]])
        legend[i[0]] = key
    return legend


def visuzalization (ldamodel, corpus, dictionary, num_words):
    viz=pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary)
    legend=topic_items(ldamodel, 15)
    
    for i, (k,v) in enumerate(legend.items()):
        plt.figure()
        plt.imshow(WordCloud(background_color="white").fit_words(ldamodel.show_topic(k, num_words)))
        plt.axis("off")
        plt.title("Topic #" + str(k+1))
        plt.show()
    
    display=pyLDAvis.display(viz)
    
    return display

def original_modeled_documents(data, dataframe, ldamodel, dictionary, num_topics):
    save = False
    
    topics = []
    exclude = set()
    
    legend=topic_items(ldamodel,15)
    
    for i,d in enumerate(data):
        document_topics=ldamodel[dictionary.doc2bow(d)]
        sorted_topics=sorted(document_topics, reverse=True, key = lambda x :x[1])[:num_topics]
        top_topics=[j for j,_ in sorted_topics]
        theme = [legend[t] for t in top_topics if t not in exclude]
        topics.append(theme)
    
    dataframe['Topics'] = topics
    
    if save:
        LL = dataframe.sort_values('Movie Name', ascending=True)
        LL.to_csv('model@O1.csv')    
    return dataframe


def data_wrangler_film2():
    import os
    path = "/Users/ViVeri/Desktop/Wave/imsdb_raw_nov_2015"
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
            text.append(inputfile.read())
            
#     tk_corpus = [i.lower().split() for i in corpus] 

    labels =[os.path.dirname(i).split('/')[-1] for i in filenames]
    
    corpus=preprocessed_data(text)

    M2=pd.DataFrame()
    M2['name']= name
    M2['genre'] = labels
    M2['text'] = text
    M2['corpus'] = corpus


    pickle.dump(M2, open( "M2.p", "wb" ) )


    return M2

def new_modeled_documents(new_data,names, labels, ldamodel, dictionary, ):
    
    save = False
    
    topics =[]
    themes = []

    exclude = set([24,21, 29,12])

    legend=topic_items(ldamodel,15)
    
    for i,d in enumerate(new_data):
        document_topics=ldamodel[dictionary.doc2bow(d)]
        sorted_documents=sorted(document_topics, reverse=True, key = lambda x :x[1])[:7]
        top_topics=[j for j,_ in sorted_documents]
        theme_id = [t for t in top_topics if t not in exclude]
        theme= [legend[t] for t in top_topics if t not in exclude]
        topics.append(theme_id)
        themes.append(theme)
        
    M=pd.DataFrame()
    M['Names'] = names
    M['Genre'] = labels
    M['Topics'] = topics
    M['Themes'] = themes

    # for i, j in enumerate(topics):
    #     M['topic' + '{:02d}'.format(i+1)] = j 
        
    
    if save:
        M.to_csv('model@.csv')
    
    return M

def load_model(model_file):
    model_=models.ldamodel.LdaModel.load(model_file)
    id2word = gensim.corpora.Dictionary()
    _ = id2word.merge_with(model_.id2word)
    
    return model_ , id2word

 
# U, M, C = datawrangler.data_wrangler('plot_summaries.txt', 'movie.metadata.tsv')
# data = datawrangler.preprocessed_data(C)
# model_, train_corpus, dictionary =datawrangler.LDA_Model(data, 5, 10)
# legend=topic_items(model_,15)
# viz =  visuzalization(model_, train_corpus, dictionary, 15)
# M=original_modeled_documents(data,M,modelo, dictionary, 5) #Matrix contains information on original documents and corresponding topics
# name, labels, corpus=data_wrangler_film2()
# R=new_modeled_documents(corpus,name,labels,modelo,id2word)# Matrix contains information on Film2.0 corpus and corresponding topics












