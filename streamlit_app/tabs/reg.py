import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

title = "Modélisation"
sidebar_name = "Modélisation"

@st.cache #(allow_output_mutation=True)
def load_data_eda1(): 
    img=Image.open('./assets/Modelisation')
    return df





def run():

    st.title(title)

    #st.markdown('Modelisation')
    
    df_1=pd.read_csv('../data/01_vgsales_clean.csv')
    df_1=df_1.sort_values(by='Global_Sales',ascending=False).iloc[1:,:]
    df_1=df_1[df_1['Global_Sales']<23].sort_values('Global_Sales',ascending=False)

    img=Image.open('./assets/Mod/machine-learning-et-big-data-660x330.jpg')
    st.image(img)
    
    st.markdown("L\'objectif de cette partie est de montrer l'amélioration de nos prédictions au fur et à mesure que nous ajoutons des features pertinentes au jeu de données de base \n") 

    st.markdown('Les modèles et métriques choisis seront : '
            '\n - **XGBoost Regressor** pour une regression avec les **métriques R² et mean_squared_error**'
            '\n - **RandomForestClassifier** pour une classification avec les **métriques accuracy et f1-score**' )

        
    #discrétisation de la variable cible

    bins=8
    target_XG=df_1['Global_Sales']
    target_RF=pd.qcut(df_1['Global_Sales'],q=bins,labels=[i for i in range(0,bins)])

    st.markdown("Selon notre modèle, la variable cible sera : \n"
           "- La variable cible **Global_sales** pour le modèle XGBoost Regressor \n "
           "- La variable cible **Global_sales** discrétisée en 8 classes pour le modèle RandomForestClassifier")
    img_6=Image.open('./assets/Mod/intervales.png')
    #qc_val=pd.qcut(df_1['Global_Sales'],q=bins).value_counts().sort_index()
    #classe_tab= pd.DataFrame(zip(qc_val.index,range(8)), columns=['Intervalle Global_Sales','Classe Global_Sales'])
    #classe_tab=classe_tab.set_index('Classe Global_Sales')
    if st.checkbox("Afficher la variable Global_Sales discrétisée"):
        st.image(img_6)
    #img=Image.open(r'C:\Users\Windows\Documents\GitHub\demo-git\streamlit_app\assets\Modelisation')
    #st.image(img)
    
    ##MODELISATION DATASET 1
    st.subheader('Modélisation avec le Dataset 1 : ')

    #df_1=pd.read_csv(r'C:\Users\Windows\Downloads\01_vgsales_clean.csv')

    st.markdown("Le jeu de données de base après avoir enlevé les features inexploitables contenait :\n"
            "- 2 features qualitatives exploitables à savoir le genre et la platforme \n"
            "- {} lignes sur {} colonnes".format(df_1.shape[0],df_1.shape[1]))

    st.dataframe(df_1.head(10))
    
    st.markdown("On transforme (OneHotEncoder) les variables qualitatives en variables quantitatives interprétables par un modèle de machine Learning" )

    df_1_b=df_1.join(pd.get_dummies(df_1[['Platform','Genre']])).drop(['Platform','Genre'],1)
    df_1_b=df_1_b.set_index('Name')

    if st.checkbox("Afficher le jeu de données transformé"):
        st.dataframe(df_1_b)
    
    st.markdown('**Résultats du modèle XGBoost Regressor sur le Dataset 1**')
    
    dr_1=pd.read_csv('./assets/Mod/data_1_XG_result.csv')
    
    st.dataframe(dr_1.set_index('Unnamed: 0'))
    
    st.markdown('**Résultats du modèle RandomForestClassifier sur le dataset 1**')
    
    RF_1=Image.open('./assets/Mod/RF_1_result.png')
    
    st.image(RF_1)
    
    
    
    
    ##MODELISATION DATASET 2
    st.subheader('Modélisation avec le Dataset 2 : ')
    
    df_2=pd.read_csv('./assets/Mod/dataset_2.csv').drop('Unnamed: 0',1)
    
    df_2_index=df_2.set_index('Name')
    
    st.markdown("Le Dataset 2 sera enrichi des variables scrapées et retravaillées suivantes : \n"
            "- Les notes sur 10 des professionnels et des joueurs \n"
            "- Le nombre de commentaires postés par les joueurs \n"
            "- Le nombre de commentaires postés par les professionnels\n")

    st.dataframe(df_2)
    
    
    st.write("Les dimensions du jeu de données avec les variables qualitatives transformées en variable quantitatives \n"
             "sont maintenant de {} lignes et de {} colonnes ".format(df_2.shape[0],df_2.shape[1]))
    
    st.markdown('**Résultats du modèle XGBoost Regressor sur le Dataset 2**')
    
    result_XG_2=pd.read_csv('./assets/Mod/data_2_XG_result.csv').set_index('Unnamed: 0')
    
    
    st.dataframe(result_XG_2)
    
    st.markdown('**Résultats du modèle RandomForestClassifier sur le Dataset 2**')
    
    RF_2=Image.open('./assets/Mod/RF_data_2.png')

    #st.markdown('### Evaluation des performances du modèle RandomForest' )
    st.image(RF_2)
    if st.checkbox("Afficher RandomForestClassifier Dataset 1"):
        st.image(RF_1)
        
    st.subheader('Modélisation avec le Dataset 3 : ')
    
    df_3=pd.read_csv(r'./assets/Mod/dataset_3.csv')
    
    st.markdown("Le Dataset 3 est enrichi avec toutes les variables scrapées et retravaillées jugées utiles : \n"
            "- Les variables logarithmiques N_user.log, N_pro.log \n"
            "- La variable cible **Global_Sales.log** \n"
            "- La variable Compound \n"
            #"- La variable Label \n"
            #"- Les années de sortie des jeux videos \n"
            "- Les franchises à succès\n"
            "- Les top éditeurs\n")
    st.markdown("Les dimensions du jeu de données complet avec le feature engineering \n"
            "sont maintenant de {} lignes et de {} colonnes ".format(df_3.shape[0],df_3.shape[1]))
    if st.checkbox("afficher la formule log"):
        st.markdown("- df['Global_Sales.log'] = np.log(df['Global_Sales'] * 1000000 + 1) \n"
                "- df['N_pro.log'] = np.log(df['N_pro']+1) \n"
                "- df['N_user.log'] = np.log(df['N_user']+1) \n")

    df_3=df_3.set_index('Name')
    
    st.dataframe(df_3)
 
    st.markdown('**Résultats du modèle XGBoost Regressor sur le Dataset 3**')
    
    result_3=pd.read_csv('./assets/Mod/data_3_XG_result.csv')
    
    st.dataframe(result_3)
    
    st.markdown('**Résultats du modèle RandomForestClassifier sur le Dataset 3**')
    
    RF_3=Image.open('./assets/Mod/RF_data_3.png')
    #st.markdown('### Evaluation des performances du modèle RandomForest' )
    st.image(RF_3)
    if st.checkbox("afficher RandomForestClassifier Dataset 2"):
        st.image(RF_2)
    
    
    
    
    
  
    
    
    
    
    
    
    
    
    
    
    
    


