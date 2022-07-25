import streamlit as st
import pandas as pd


title = "Analyse de sentiments"
sidebar_name = "Analyse de sentiments"

@st.cache #(allow_output_mutation=True)
def load_data_nlp(): 
    df_head = pd.read_csv('../streamlit_app/assets/nlp/df_head.csv')
    meta_desc= pd.read_csv('../streamlit_app/assets/nlp/meta_describe.csv')
    df_clean= pd.read_csv('../streamlit_app/assets/nlp/df_clean.csv', index_col=[0])
    #df_strat
    return df_head, meta_desc, df_clean


bow_dict={'Decision Tree Classifier':'../streamlit_app/assets/nlp/DecisionTreeClassifier_acc_0.18.png', 
          'Logistic Regression':'../streamlit_app/assets/nlp/LogisticRegression_acc_0.28.png',
          'AdaBoost Classifier': '../streamlit_app/assets/nlp/AdaBoostClassifier_acc_0.22.png', 
          'DNN 5-layers':'../streamlit_app/assets/nlp/BoW-DNN_acc_0.52.png'}

emb_dict={'Decision Tree Classifier':'../streamlit_app/assets/nlp/DecisionTreeClassifier_acc_0.22.png', 
          'Logistic Regression':'../streamlit_app/assets/nlp/LogisticRegression_acc_0.29.png',
          'AdaBoost Classifier': '../streamlit_app/assets/nlp/AdaBoostClassifier_acc_0.23.png', 
          'DNN 5-layers':'../streamlit_app/assets/nlp/WE-DNN_acc_0.37.png'}

def run():

    st.title(title)
    
    df_head, meta_desc, df_clean = load_data_nlp()
    
    st.image('../streamlit_app/assets/nlp/video-games.gif')

    st.markdown(
        """
        ### 1- Chargement du dataset
Pour cette partie du projet nous utiliserons les commentaires et notes des utilisateurs récupérées sur le site Metacritic (scraping).

Le dataset contient 1 624 963 lignes et est composé de 7 colonnes :

\t- Name : Le nom du jeu - type object
\t- Platform : La plateforme du jeu - type object
\t- Genre : Le genre du jeu - type object
\t- Date_crit : La date de la critique - type object
\t- Name_crit : Le nom de l'utilisateur - type object
\t- Score_crit : Le score donné par l'utilisateur (entier compris 0 et 10) - type int
\t- Comment_crit : Le commentaire laissé par l'utilisateur - type object

Le dataset ne contient pas de valeurs manquantes.

Nous nous intêresserons ici uniquement au lien entre le commentaire laissé par l'utilisateur (Comment_crit) et le score (Score_crit).

Score_crit sera la variable cible et il faudra donc transformer les commentaires (Comment_crit) en 'features' à l'aide des outils de Natural Language Processing (NLP) afin pouvoir utiliser des modèles de classification pour prédire un score utilisateur en fonction d'une critique.

Cela nous servira par la suite à prédire le score d'un jeu pas encore sorti en fonction des critiques/commentaires disponibles en ligne (Reddit, Twitter, etc...).
        """
    )        
        
    st.dataframe(df_head)
    
    st.markdown(
        """
        ### 2- Ajout des méta-données

On extrait les méta-données des critiques ce qui se traduit par l'ajout des colonnes :

\t - links : Nombre de liens url présents dans la critique
\t - mails : Nombre d'adresses email présentes dans la critique
\t - quotes : Nombre de citations présentes dans la critique
\t - hashtags : Nombre de hashtags présents dans la critique
\t - capslock : Nombre de majuscules présentes dans la critique
\t - chain_capslock : Nombre de majuscules à la suite présentes dans la critique
\t - exclamations : Nombre de points d'exclamation présents dans la critique
\t - chain_exclamation : Nombre de points d'exclamation à la suite présents dans la critique
\t - interogation : Nombre de points d'interrogation présents dans la critique
\t - etc : Nombre de ... présents dans la critique
\t - nb_caracter : Nombre de caractères présents dans la critique
\t - nb_words : Nombre de mots présents dans la critique
\t - nb_sentences : Nombre de phrases présents dans la critique   

Inspectons les valeures moyennes des méta-données pour chaque label/score :          
        """
    )
        
    st.dataframe(meta_desc)

    st.markdown(
        """
### 3- Nettoyage des critiques

Maintenant que nous avons extrait les méta-données de nos critiques, nous pouvons procéder au nettoyage de celles-ci.
Lors de ce traitement, en on profitera pour supprimer les éléments suivants:
            
\t- les tags html
\t- les caractères qui ne sont pas des lettres
\t- les liens url
\t- les adresses email
\t- les hashtags
\t- les citations
\t- la ponctuation
\t- transformation de tous les caractètres en minuscules
\t- suppression des soptwords (anglais)
          
Nous pouvons comparer ci-dessous le text avant et après nettoyage :  
        """
    )

    st.dataframe(df_clean)

    st.markdown(
    """
    ### 4- Affichage d'un nuage de mots

Avant d'aller plus en avant, faisons une représentation sous forme de nuage de mots de notre corpus de critiques.

Le dataset étant très volumineux (+ de 1.5 millions de lignes) nous allons effectuer le nuage de mot que sur une portion de ce dernier (10%)
    """
)
    st.image('../streamlit_app/assets/nlp/wordcloud.png')

        
    st.markdown(
    """
    ### 5- Réduction et ré-équilibrage du jeu de données

    Le jeu de données étant volumineux nous allons procéder à une première réduction de celui-ci.
    Pour cela nous ne conserverons que les lignes pour lesquelles le nombre de mots est inférieur ou égal à 500 (le nombre de mots moyen étant compris entre 70 et 158 pour chaque label, cf. description statistique des méta-données).
    
    Observons la distribution des scores au sein du dataframe :
    """
)
        
    st.image('../streamlit_app/assets/nlp/distrib_score.png')
    
    st.markdown(
    """
On remarque que les labels ne sont pas distribués de manière uniforme et nous allons donc effectuer un ré-équilibrage/ré-échantillonage du dataframe en prennant un échantillon de 40 000 lignes par label (ce qui correspond +/- au nombre de ligne du label le moins représenté - le 2).

Ce qui nous donne après ré-échantillonnage la distribution suivante :
    """
)
        
    st.image('../streamlit_app/assets/nlp/distrib_score_eq.png')
    
    st.markdown(
    """
    Nous séparons ensuite le jeu de données ré-équilibré en jeu d'entrainement et jeu de test (80/20 %)à l'aide de la méthode train_test_split.
    Nous en profitons également pour normaliser les valeurs des méta-données à l'aide la méthode StandardScaler.
    """
)    
    st.markdown(
    """
    ### 6- Prédiction à l'aide d'un modèle Bag of Words

Afin de convertir nos critiques en vecteurs nous utiliserons ici la méthode TfidfVectorizer pour laquelle nous choisirons les paramètres ci-dessous:

\t- min_df = 5, ignore les termes peu fréquents (dans moins de 5 docs)
\t- ngram_range=(1,3), prend en compte les 1-grams, 2-grams et 3-grams
\t- max_features= 2000, dimensionalité du vecteur de mots

Une fois nos critiques sous forme de vecteur nous allons tester plusieurs modèles de classification :

\t- Decision Tree Classifier
\t- Logistic Regression
\t- AdaBoost Classifier
\t- DNN 5-layers
    """
)
    option_bow = st.selectbox(
     'Choix du model: ',
    ('Decision Tree Classifier', 'Logistic Regression','AdaBoost Classifier','DNN 5-layers'),index=0)
    
    st.image(bow_dict[option_bow])
    
    st.markdown(       
    """
    On remarque ici que le modèle DNN 4-layers donne les meilleurs bien que les métriques ne soient pas très élevées.
    On constate cependant, en inspectant la matrice de confusion que les prédictions ne sont pas si mauvaises.
    Le manque de précision pouvant s'expliquer sur le nombre de labels à prédire (11) ainsi que sur la nature des features (text).'
    """
   ) 
    
    st.markdown(
    """
    ### 7- Prédiction à l'aide d'un modèle de Word Embedding
Afin de vectoriser nos critiques à l'aide d'un modèle de Word Embedding nous allons créer notre propre modèle de Word Embedding plûtot que d'en utiliser un déjà pré-entrainé.
Pour cela, nous commençons par concaténer toutes les phrases de notre corpus de critiques dans une liste.
Ensuite, nous utiliserons la méthode Word2Vec de la bibliothèque Gensim en lui spécifiant les paramètres ci-dessous :
    
\t- size = 2000, Dimensionalité du vecteur de mots
\t- min_count = 10, ignore les mots dont la fréquence est inférieure à 10
\t- window = 4, distance maximum entre le mot et le mot à prédire dans une phrase
\t- sample = 1e-3, seuil de ré-échantillonnage (baisse) pour les mots très fréquents

Une fois le modèle créé nous pouvons l'utiliser pour calculer la moyenne des mots de chacune de nos critiques et ainsi obtenir une représentation vectorielle de celles-ci.

A l'instar de ce que nous avons fait précedement pour le modèle Bag of Words nous allons pouvoir en faire de même avec notre modèle Word Embedding en testant à nouveau nos modèles de classification.

\t- Decision Tree Classifier
\t- Logistic Regression
\t- AdaBoost Classifier
\t- DNN 5-layers
    """
    )
        
    option_emb = st.selectbox(
     'Choix du model: ',
    ('Decision Tree Classifier', 'Logistic Regression','AdaBoost Classifier','DNN 5-layers'),index=1)
    
    st.image(emb_dict[option_emb])
    
    st.markdown(       
    """
    Encore cette fois, c'est le model DNN 5-layers qui donne les meilleurs résultats toutefois ceux-ci sont largement en retrait par rapport aux résultats obtenus avec le Bag of Ward.
    Nous opterons donc pour le model DNN 5-layers à base de Bag of Word afin d'estimer le score des jeux à venir dans la partie suivante.
    """
   ) 

#    st.markdown(       
#    """
#    ### 7- Prédiction à l'aide de la combinaison des modèles BoW et Word Embedding
#Nous allons dans cette partie récupérer les vecteurs de probabilités retournés par nos modèles DNN 4-layers par BoW et Word Embedding et les utiliser comme features pour entrainer à nouveau des modèles de classification.
#    """
#   ) 
    
#    option_combo = st.selectbox(
#     'Choix du model: ',
#    ('Decision Tree Classifier', 'Logistic Regression','AdaBoost Classifier','DNN 4-layers'),index=2)
    
#    st.image(bow_dict[option_combo])