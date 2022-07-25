#######################################################
############### Importations de packages ##############
#######################################################

import pandas as pd  # Importation de pandas sous l'alias pd
import numpy as np  # Importation de Numpy sous l'alias np
import matplotlib.pyplot as plt  # Importation de Matplotlib sous l'alias plt
import seaborn as sns  # Importation de seaborn sous l'alias sns

from tqdm import tqdm  # Importation de tqdm
from joblib import dump, load

import nltk
import re  # Importation de re
from bs4 import BeautifulSoup 
from nltk.corpus import stopwords
from gensim.models import KeyedVectors

# Outils scikit learn
from sklearn.metrics import classification_report, confusion_matrix

# Chargement de tensorflow, keras etc...
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

# Gestion mémoire graphique keras-gpu
from keras.backend import set_session, clear_session, get_session
import gc

# Chargement de warnings pour gérer les avertissements
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


#######################################################
######### Fonctions de pré-processing du text #########
#######################################################

# Suppression des liens urls
def removeLink(text):
    r1 = re.compile(r"https?://[a-zA-Z0-9./]+")
    links = r1.findall(text)
    text = r1.sub('', text)
    r2 = re.compile(r"www\.[a-zA-Z0-9.-:/]+")
    links += r2.findall(text)
    return r2.sub('', text)

# Suppression des adresses email
def removeMail(text):
    r = re.compile(r"[a-zA-Z0-9.-]+@[a-zA-Z.]+")
    mails = r.findall(text)
    return r.sub('', text)

# Remplacement des hashtags
def replaceHashtag(text):
    r = re.compile(r"#[a-zA-Z0-9]+")
    hashtag = r.findall(text)
    text = text.replace('#', '_')
    return r.sub('_', text)

# Suppression des citations
def removeQuote(text):
    r = re.compile(r"@[a-zA-Z0-9]+")
    quote = r.findall(text)
    return r.sub('', text)

# Recherches du nombre de liens urls
def findLink(text):
    r1 = re.compile(r"https?://[a-zA-Z0-9./]+")
    links = r1.findall(text)
    r2 = re.compile(r"www\.[a-zA-Z0-9.-:/]+")
    links += r2.findall(text)
    return len(links)

# Recherches du nombre d'adresses email
def findMail(text):
    r = re.compile(r"[a-zA-Z0-9.-]+@[a-zA-Z.]+")
    mails = r.findall(text)
    return len(mails)

# Recherches du nombre de citations
def findQuote(text):
    r = re.compile(r"@[a-zA-Z0-9]+")
    quote = r.findall(text)
    return len(quote)

# Recherches du nombre de hashtags
def findHashtag(text):
    r = re.compile(r"#[a-zA-Z0-9]+")
    hashtag = r.findall(text)
    return len(hashtag)

# Recherches du nombre de caractères en majuscules
def findCAPSLOCK(text):
    r = re.compile(r"[A-Z]")
    capslock = r.findall(text)
    return len(capslock)

# Recherches du nombre de chaines de caractères en majuscules
def find_chain_CAPSLOCK(text):
    r = re.compile(r"[A-Z]{2,}")
    capslock = r.findall(text)
    return len(capslock)

# Recherche du nombre de points d'exclamation
def find_exclamation(text):
    r = re.compile(r"\!")
    exclamation = r.findall(text)
    return len(exclamation)

# Recherche du nombre de chaines de points d'exclamation
def find_chain_exclamation(text):
    r = re.compile(r"\!{2,}")
    exclamation = r.findall(text)
    return len(exclamation)

# Recherche du nombre de points d'interrogation
def find_interogation(text):
    r = re.compile(r"\?")
    interogation = r.findall(text)
    return len(interogation)

# Recherche du nombre de '...'
def find_etc(text):
    r = re.compile(r"\.{2,}")
    etc = r.findall(text)
    return len(etc)


#######################################################
############# Calcule des Metadonnées #################
#######################################################

# Calcule et ajout des métadonnées au dataframe
def RetrieveMetaData(df,col_name):
    # Ajout des metas données du texte
    tqdm.pandas(desc='Ajout de la colonne links')
    df['links']= df[col_name].progress_apply(lambda x: findLink(x)) # Nombre de liens urls
    tqdm.pandas(desc='Ajout de la colonne mails')
    df['mails']= df[col_name].progress_apply(lambda x: findMail(x)) # Nombre d'adresses email
    tqdm.pandas(desc='Ajout de la colonne quotes')
    df['quotes']= df[col_name].progress_apply(lambda x: findQuote(x)) # Nombre de citations
    tqdm.pandas(desc='Ajout de la colonne hashtags')
    df['hashtags']= df[col_name].progress_apply(lambda x: findHashtag(x)) # Nombre de hashtags
    tqdm.pandas(desc='Ajout de la colonne capslock')
    df['capslock']= df[col_name].progress_apply(lambda x: findCAPSLOCK(x)) # Nombre de majuscules
    tqdm.pandas(desc='Ajout de la colonne chain_capslock')
    df['chain_capslock']= df[col_name].progress_apply(lambda x: find_chain_CAPSLOCK(x)) # Nombre de suite de majuscules
    tqdm.pandas(desc='Ajout de la colonne exclamations')
    df['exclamations']= df[col_name].progress_apply(lambda x: find_exclamation(x)) # Nombre de !
    tqdm.pandas(desc='Ajout de la colonne chain_exclamation')
    df['chain_exclamation']= df[col_name].progress_apply(lambda x: find_chain_exclamation(x)) # Nombre de ! à la suite
    tqdm.pandas(desc='Ajout de la colonne interogation')
    df['interogation']= df[col_name].progress_apply(lambda x: find_interogation(x)) # Nombre de ?
    tqdm.pandas(desc='Ajout de la colonne etc')
    df['etc']= df[col_name].progress_apply(lambda x: find_etc(x)) # Nombre de '...'
    tqdm.pandas(desc='Ajout de la colonne nb_caracter')
    df['nb_caracter'] = df[col_name].progress_apply(len) # Longueur de la chaine de caractères
    tqdm.pandas(desc='Ajout de la colonne nb_words')
    df['nb_words']= df[col_name].progress_apply(lambda x: len(re.sub(r'[^\w\s]','',x).split())) # Nombre de mots
    tqdm.pandas(desc='Ajout de la colonne nb_sentences')
    df['nb_sentences']= df[col_name].progress_apply(lambda x:len(re.split(r'[!?]+|(?<!\.)\.(?!\.)', x.replace('\n',''))[:-1]))
    return df


#######################################################
############### Découpage phrases/Mots ################
#######################################################

# Fonction séparant un commentaire en suite de phrases
def review_sentences(review, tokenizer, remove_stopwords=True):
    # 1. Utilisation du tokenizer de nltk
    raw_sentences = tokenizer.tokenize(review.strip())
    sentences = []
    # 2. Boucle for sur chaque phrase
    for raw_sentence in raw_sentences:
        if len(raw_sentence)>0:
            sentences.append(review_wordlist(raw_sentence,\
                                            remove_stopwords))
    # Retourne une liste de listes 
    return sentences

# Fonction convertissant le text en une suite de mots
def review_wordlist(review, remove_stopwords=True):
    # 1. Suppression de tags html à l'aide de BeautifulSoup
    review_text = BeautifulSoup(review).get_text()
    # 2. Suppression des caracères qui ne sont pas des lettres
    review_text = re.sub("[^a-zA-Z]"," ",review_text)
    # 4. Suppresion des liens url
    review_text= removeLink(review_text)
    # 5. Suppression des adresses emails
    review_text= removeMail(review_text)
    # 6 Suppression des hashtags
    review_text= replaceHashtag(review_text)
    # 7. Suppression des citations
    review_text= removeQuote(review_text)
    # 8. Suppression de la ponctuation
    review_text= re.sub(r'[^\w\s]','', review_text)
    # 9. Conversion en caractères minuscules
    words = review_text.lower().split()   
    # 10. Suppression des stopwards (par défaut activé)
    if remove_stopwords:
        stops = set(stopwords.words("english"))     
        words = [w for w in words if (not w in stops) & (len(w)>1)]
    # retourne la liste de mots nettoyée
    return(words)


#######################################################
################### Word Embedding ####################
#######################################################

# Function qui retourne le vecteur correspondant à la somme des vecteurs de la critique
def featureVecMethod(words, model, num_features):
    # Pre-initialising empty numpy array for speed
    featureVec = np.zeros(num_features,dtype="float32")
    nwords = 0
    
    #Converting Index2Word which is a list to a set for better speed in the execution.
    index2word_set = set(model.index_to_key)
    
    for word in  words:
        if word in index2word_set:
            nwords = nwords + 1
            featureVec = np.add(featureVec,model[word])
    
    # Dividing the result by number of words to get average
    #if nwords != 0:
    #    featureVec = np.divide(featureVec, nwords)
    return featureVec

# Fonction qui calcule la matrice de Word Embedding
def get_weight_matrix(emb_model, all_words, vocab_size):
    # define weight matrix dimensions with all 0
    weight_matrix = np.zeros((vocab_size, emb_model.vector_size))
    # step vocab, store vectors using the Tokenizer's integer mapping
    for i in tqdm(range(len(all_words)),desc='Création de la matrice d\'Embedding'):
        if all_words[i] in emb_model.index_to_key:
            weight_matrix[i + 1] = emb_model[all_words[i]]
    return weight_matrix


#######################################################
################## Echantillonnage ####################
#######################################################

# Fonction d'échantillonage/équilibrage du dataset
def stratified_sample(df, col, n_samples):
    n = min(n_samples, df[col].value_counts().min())
    df_ = df.groupby(col).apply(lambda x: x.sample(n))
    df_.index = df_.index.droplevel(0)
    return df_


#######################################################
############ Affichage résultats classif ##############
#######################################################

# Fonction d'affichage du rapport de classification et Matrice de confusion
def display_results(model,xtest,ytest,oh=False):
    name= type(model).__name__
    y_pred= model.predict(xtest)
    
    if oh:  # Dans le cas d'un model DeepLearning avec one-hot encoding des labels
        name= model._name
        y_pred= np.argmax(y_pred,axis=1)
        ytestoh= to_categorical(ytest, dtype = 'int')
        score= model.evaluate(xtest,ytestoh, verbose= 0)[1]
    else:  # Model de ML classiques
        name= type(model).__name__
        score= model.score(xtest, ytest)
    
    # Calcule du rapport de classification
    clf_report = classification_report(y_pred, ytest, output_dict=True)
    # Calcule de la matrice de confusion
    cnf_matrix= confusion_matrix(y_pred,ytest)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={'width_ratios': [1, 3]})
    fig.suptitle('Résultats du model '+name+' - Accuracy (test): '+str(round(score,2)), fontsize=15)

    # Affichage du Classification report
    sns.heatmap( pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True, cmap= 'GnBu', cbar=False,ax=axes[0])
    axes[0].set_title('Rapport de classification')

    # Affichage de Confusion matrix
    s= sns.heatmap(cnf_matrix, annot=True, fmt=".0f",cmap= "rocket_r",ax=axes[1]) 
    s.set(xlabel='Score prédit', ylabel='Score réel')
    axes[1].set_title('Matrice de confusion')
    # Sauvegarde de la figure
    filename= name+'_'+'acc_'+str(round(score,2))+'.png'
    plt.savefig(filename)
    

#######################################################
############ Affichage résultats classif ##############
#######################################################   

# Gestion de la mémoire vive pour keras-gpu
def reset_keras():
    sess = get_session()
    clear_session()
    sess.close()
    sess = get_session()
    print(gc.collect()) # RAM libérée

    # Conserve la session utilisée au départ
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 1
    config.gpu_options.visible_device_list = "0"

