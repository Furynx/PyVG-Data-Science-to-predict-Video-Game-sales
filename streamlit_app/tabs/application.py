import streamlit as st
import pandas as pd


title = "Mise en application"
sidebar_name = "Mise en application"

@st.cache #(allow_output_mutation=True)
def load_data_nlp(): 
    df_head = pd.read_csv('../streamlit_app/assets/nlp/df_head.csv')
    meta_desc= pd.read_csv('../streamlit_app/assets/nlp/meta_describe.csv')
    df_clean= pd.read_csv('../streamlit_app/assets/nlp/df_clean.csv', index_col=[0])
    #df_strat
    return df_head, meta_desc, df_clean


user_sa={'Bag of Word - Logistic Regression':'../streamlit_app/assets/nlp/bow_logreg.png',
          'Bag of Word - AdaBoost Classifier': '../streamlit_app/assets/nlp/bow_ada.png', 
          'Bag of Word - DNN':'../streamlit_app/assets/nlp/bow_dnn.png',
          'Word Embedding - Logistic Regression':'../streamlit_app/assets/nlp/emb_logreg.png',
          'Word Embedding - AdaBoost Classifier': '../streamlit_app/assets/nlp/emb_ada.png',
          'Word Embedding - Conv1D':'../streamlit_app/assets/nlp/emb_conv1d.png'}


def run():

    st.title(title)

    df_head, meta_desc, df_clean = load_data_nlp()
    
    #<p class="aligncenter"><img src='https://thumbs.gfycat.com/VapidFairHyracotherium-size_restricted.gif'/></p>
    
    st.markdown(    
        """
        <style>
        .aligncenter { text-align: center;}
       
        </style>      
        
        <h3>Objectif</h3>
        
        <p>Dans cette partie notre objectif va être d'utiliser les modèles retenus précédement afin de déterminer les ventes globales d'un jeu à venir.<p/>
        
        <p>Pour notre exemple nous avons choisi le jeu <b>Diablo IV</b> dont la sortie initialement prévue pour fin 2022 a été repoussée à 2023.</p>
        
        <p class="aligncenter"><img src='https://media0.giphy.com/media/XYBIGvYOaOfqE/giphy.gif'/></p>
        
        <p>Une partie des données (Nom, Genre, Editeur, Développeur ...) concernant le jeu est disponible et sera récupérée en scraping sur les sites VGChartz et Metacritic.</p>
        
        
        <table align="center">
        <tr>
            <td><img src="https://www.guinnessworldrecords.com/Images/vgchartz-760x444_tcm25-607263.jpg?ezimgfmt=rs:412x241/rscb1/ng:webp/ngcb1" alt="" width='150'/></td>
            <td><img src="https://seekvectorlogo.com/wp-content/uploads/2020/06/metacritic-vector-logo.png" alt="" width='150'/></td>
        </tr>
        </table> 
        <br/>
        
        <p>Toutefois certaines des features utilisées par nos modèles de prédiction ne sont connues qu'après la sortie du jeu.</p>
        
        <div>C'est le cas des variables :
            <li><b>Score_user</b> : moyenne des notes données par les utilsateurs de Metacritic</li>
            <li><b>N_user</b> : nombre de critiques utilisateurs de Metacritic</li>
            <li><b>Score_pro</b> : moyenne des notes données par les professionnels sur Metacritic</li>
            <li><b>N_pro</b> : nombre de critiques de professionnels sur Metacritic</li>
            <li><b>compound</b> : score d'analyse de sentiment calculé à l'aide de la bibliothèque VaderSentiment depuis les commentaires utilisateurs de Metacritic</il>
        </div>
        <p></p>
        
        <p>Nous allons de ce fait devoir estimer ou choisir arbitrairement les valeurs de ces features.</p>
        
        
        <h4>Détermination de Score_user, N_user et compound</h4>
        
        <p>Pour palier au fait qu'aucune critique/score ne soit disponible sur Metacritic avant la sortie du jeu nous allons ici contourner le problème en récupérant les commentaires et discussions à propos de celui-ci depuis le forum Reddit (SubReddit) dédié à Diablo IV.</p>
        
        <p>Nous utiliserons pour cela l'API PRAW (Python Reddit API)
        
        <p class="aligncenter"><img src='https://stuff.co.za/wp-content/uploads/2021/09/reddit_logo_main.jpg' width='150'/></p>
        
        <div>
        
        <li>On pourra dès lors considérer que <b>N_user</b> correspond au nombre de commentaires récupérés sur le SubReddit du jeu.</li>
        <br/>
        <li>Puis on utilisera la bibliothèque VaderSentiment sur les commentaires du SubReddit afin d'estimer un score de <b>compound</b></li>
        <br>
        <li>Pour effectuer une estimation du <b>Score_user</b> nous avons implémenté un modèle d'analyse de sentiments basé sur les critiques utilisateurs de Metacritic.
        
        <p>Pour la détermination de ce modèle nous avons été amenés à pré-traiter nos données (récupération métadonnées, suppression des liens, hashtags, emails, stopwords ...) tester différentes approches (Bag of Word, Word Embedding) ainsi que différents modèles (Régression Logistique, AdaBoost, Réseau de neurones).</p>
        """, unsafe_allow_html=True) 
                

    option_sa = st.selectbox('Résultats des modèles testés : ',
        ('Bag of Word - Logistic Regression',
         'Bag of Word - AdaBoost Classifier', 
         'Bag of Word - DNN',
         'Word Embedding - Logistic Regression',
         'Word Embedding - AdaBoost Classifier',
         'Word Embedding - Conv1D')
        ,index=0)
        
    st.image(user_sa[option_sa])
        
    st.markdown(
         """  
        <p>C'est l'utilisation du Word Embedding avec un modèle à base de couche de neurones Conv1D qui s'est révèlé le plus perfomant lors de nos tests.</p>
        </li>
        
        </div>
        
        <h4>Détermination de Score_pro, N_pro</h4>
        
        <p>Dans la mesure où l'on a pu observer des corrélations existantes avec nos autres features et avec la variable <b>Score_pro</b> on utilisera ici un modèle XGBoostRegressor afin d'en obtenir une estimation.</p>
        """, unsafe_allow_html=True) 
        
    st.image('../streamlit_app/assets/nlp/pro_corr.png',  width= 500)
    st.image('../streamlit_app/assets/nlp/score_pro.png')
        
    st.markdown(
         """          
        <p>En ce qui concerne la variable <b>N_pro</b> nous avons choisi de lui affecter la valeur arbitraire de <b>1</b> afin de minimiser son importance du fait que le Score_pro soit une estimation.</p>
        
        <h4>Prédictions des ventes pour Diablo IV</h4>
        
        <p>Maintenant que nous disposons de toutes les données nécessaires, nous allons pouvoir effectuer les prédictions avec nos modèles de régression et de classification.
        
        <h5>Estimation des ventes par régression</h5>
        
        <p>Pour rappel, le modèle de régression retenu est un modèle XGBoost Regressor. </p>
        
        <p>La prédiction à l'aide de ce dernier nous retourne les résultats ci-dessous :</p>
        """, unsafe_allow_html=True) 
        
    st.image('../streamlit_app/assets/nlp/d4_reg.png')
        
    st.markdown(
            """
        <p><b>Analyse des résultats obtenus : </b>
        La prédiction obtenue par notre modèle semble bien en-deça de ce à quoi on peut s'attendre d'une licence de jeu AAA développée par une compagnie telle que Activision Blizzard.</p>

        <p>Il est probable que le modèle n'a pas su ici capter les éléments nécessaires pour faire une estimation plus fine du Global_Sales.</p>

        <p>Cela peut s'expliquer par des données de qualités insuffisantes ainsi que du manque de features pertinentes. En effet, le regroupement de nos datasets a entrainé la perte de certaines entrées pour les jeux de la même franchise (Diablo) tels que Diablo ou encore Diablo III qui représentent des millions de ventes toutes plateformes confondues.</p>

        <p>On pourrait donc s'attendre à un résultat beaucoup plus élevé.</p>
        
        <h5>Estimation des ventes par classification</h5>
        
        <p>Le modèle retenu pour la classification est le RandomForest Classifier. </p>
        
        <p><i>Pour rappel : </i>Nous avons découpé les ventes globales en 8 classes et dont la correspondance est récapitulée dans le tableau ci-dessous : </p>
        """, unsafe_allow_html=True) 
        
    st.image('../streamlit_app/assets/nlp/intervales.png')
        
    st.markdown(
            """        
        <p>Notre prédiction pour Diablo IV nous retourne alors les résultats suivants :</p>
        """, unsafe_allow_html=True) 
        
    st.image('../streamlit_app/assets/nlp/d4_class.png')
        
    st.markdown(
            """        
        <p><b>Analyse des résultats obtenus : </b>
        Les résultats obtenus avec la classification sont plus bien meilleurs que ceux obtenus précédemment en régression toutefois l'intervalle des ventes globales toutes Plateformes confondues est très large (~3.5m à ~68m) ce qui ne permet pas d'avoir une estimation très précise.</p>
        """
    , unsafe_allow_html=True)  
    
