import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import io
import plotly.express as px
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA

title = "Etape 1: Constituer le jeu de données"
sidebar_name = "Dataset"

@st.cache #(allow_output_mutation=True)
def load_data_eda1(): 
    df = pd.read_csv('../data/vgsales.csv')
    return df

@st.cache #(allow_output_mutation=True)  
def load_data_eda2(): 
    df = pd.read_csv('../data/vgsales_eda2.csv')
    return df
    
@st.cache #(allow_output_mutation=True)  
def load_data_eda3(): 
    df = pd.read_csv('../data/vgsales_eda2_ready.csv')
    return df    
    

def run():

    st.title(title)
    df = load_data_eda1()
    
    st.markdown("### 1- Affichage du dataset initial VGSales.csv")
    st.markdown(f"""
    > La taille du jeu de données est de __{df.shape[0]}__ réparties sur __11__ variables/colonnes, chaque ligne représente un couple jeu/plateforme sorti entre 1980 et 2016.""")
    fig = px.histogram(df, x="Year",title="Distribution des jeux sortis par année",
                        width=600, height=400)
    fig.update_layout(
        margin=dict(l=20, r=10, t=40, b=20),
        paper_bgcolor="LightSteelBlue",
    )
    st.plotly_chart(fig, use_container_width=True, )
    
    st.markdown("__Description du dataset:__")
    st.dataframe(pd.read_csv('../data/info.csv' ),height=350)

    st.markdown("__Remarques principales:__")
    st.markdown("""
    > - Très peu de variables explicatives: plateforme, année de sortie, genre et éditeur.
    > - Exclusivement des variables explicatives catégorielles (la date n'exprime pas une quantité).
    > - Absence de données pour les plateformes actuelles (Switch, PS5, Xbox One…).
    > - Peu de valeurs manquantes
    """)

    st.markdown("### 2- Collecte de nouvelles données")

    st.markdown("""
    - Principaux sites web de réference du jeux video: jeuxvideo.com, metacritics, VGChartz, IGDB, RAWG, HowLongToBeat, Youtube, Twitch, Wikipedia
    - Au total, plus de __9__ sites web ont été scrapés avec BeautiFullSoup, Selenium ou via des API
    __Classement des informations utiles par categorie:__
    """)
    df_scraping = pd.read_csv('assets/info_scraping.csv',sep=';')
    st.dataframe(df_scraping,height=500,width=None)
    
    st.markdown("""__Sources retenues pour le projet:__""")

    st.image("assets/merge.png",width=700)

    buffer = io.StringIO()
    df2 = load_data_eda2()
    df2.info(buf=buffer)
    s = buffer.getvalue()

    #st.code(s)
    #st.markdown(df.info())

    st.markdown("""__Utilisation d'une clé de jointure:__""")
    st.code("""df['game_key'] = df['Name'].str.normalize('NFKD')
                           .str.encode('ascii',errors='ignore')
                           .str.decode('utf-8')
                           .str.replace('[^A-Za-z0-9]+', '-', regex=True)
                           .str.replace('^-+|-+$', '', regex=True)
                           .str.lower()""",language='python')
    
    
    col1, col2 = st.columns(2)
    col1.markdown("""Disney's Lilo & Stitch :arrow_right: disney-s-lilo-stitch""")
    col1.markdown("""Super Monkey Ball: Step & Roll :arrow_right: super-monkey-ball-step-roll""")
    col2.metric("Row", "8938", delta="200%")
    
    st.markdown("### 3- Feature engineering")
    
    st.markdown("""__A- Simplification des plateformes:__""")
    st.image("assets/console_gen.png") 
    st.code("""def OEM(r):
    oem = 'Others'
    if r['Platform'] in ['X360',  'XB', 'XOne']:
        oem = "xbox"
    elif r['Platform'] in ['PS', 'PS2', 'PS3', 'PS4', 'PSP', "PSV"]:
        oem = "playstation"
    elif r['Platform'] in ['Wii', 'WiiU', 'N64', 'GC', 'NS', "3DS", 'DS']:
        oem = "nintendo"
    elif r['Platform'] in ['PC']:
        oem = "PC"
    return oem""")
    
    st.markdown("""__B- Normalisation de la distribution des données:__""")
    st.code("""# https://dataanalyticspost.com/Lexique/loi-gaussienne/
# https://datascientest.com/courbe-gaussienne
def normalise_log(df):
    df['Global_Sales.log'] = np.log(df['Global_Sales'] * 1000000 + 1)
    df['N_pro.log'] = np.log(df['N_pro']+1)
    df['N_user.log'] = np.log(df['N_user']+1)
    return df""")

    st.markdown("""__C- Clustering (KMeans):__""")
    st.image("assets/coude.png") 
    st.image("assets/PCA.png")
    st.code("df['labels'] = labels")
    
    st.markdown("""__D- Top franchises:__""")
    st.markdown("""Franchises à succès.""")
    st.code("""franchise_list = df.Franchise_wikipedia.value_counts().head(12).index.tolist()
df['Franchise_top'] = df.apply(lambda r: r['Franchise_wikipedia'] if r['Franchise_wikipedia']
                               in franchise_list else 'Others', axis=1)
df = df.join(pd.get_dummies(df.Franchise_top,drop_first=True,prefix='lic'))""")

    st.markdown("""__E- Top éditeurs:__""")
    st.markdown("""Editeurs dominants supposés avoir le plus de moyens et de succès en général.""")
    st.code("""publisher_list = df.Publisher.value_counts().head(12).index.tolist()
df['Publisher_top'] = df.apply(lambda r: r['Publisher'] if r['Publisher']
                               in publisher_list else 'Others', axis=1)
df = df.join(pd.get_dummies(df.Publisher_top,drop_first=True,prefix='pub'))""")

    st.markdown("### 4- Valeurs aberrantes")
    st.markdown("""> Utilisation de l'algorithme non-supervisé IsolationForest
>
> C'est une étape importante dans le cas de la régression qui donne de meilleurs résultats quand appliquée""")
    

    st.code("""isof = IsolationForest(contamination=0.02, n_estimators=100)
isof.fit(df.select_dtypes(exclude='object'))
df['anom'] = isof.predict(df.select_dtypes(exclude='object'))
df = df.drop(df[df['anom']<0].index)
df = df.drop(columns=['anom'])""")
    
    st.markdown("### 5- PCA 3D")
    st.markdown("Les points en couleur représentent la valeur cible Global Sales (bins=8)")


    df3 = load_data_eda3()
    col_ = [#'Score_pro', 'Score_user',
            'N_pro', 'N_user',
            'compound',
            'N_pro.log', 'N_user.log',
            'year',
            'compound',
            'PC', 'nintendo', 'playstation', 'xbox',
            #'labels'
           ]
           
    # 
    target = pd.qcut(df3['Global_Sales'],q=8)
    features = df3[col_].copy()
    
    scaler = StandardScaler() 
    scaled_df = pd.DataFrame(scaler.fit_transform(features), columns=col_)
    pca = PCA (n_components=0.9)
    data_2D = pca.fit_transform(scaled_df)
    oem_enc = LabelEncoder()
    x = oem_enc.fit_transform(target)

    df4 = pd.DataFrame({'x':data_2D[:, 0],'y':data_2D[:, 1],'z':data_2D[:, 3],'target':x})
    df4["target"] = df4["target"].astype(str)

    fig = px.scatter_3d(x=df4.x, y=df4.y, z=df4.z, color=target, title="Données projetées sur les 3 axes de PCA",opacity = 0.8, width=800, height=800,
    labels={
                     "x": f'PCA1 {np.round(pca.explained_variance_ratio_[0],2)}%',
                     "y": f'PCA2 {np.round(pca.explained_variance_ratio_[1],2)}%',
                     "z": f'PCA3 {np.round(pca.explained_variance_ratio_[2],2)}%'
                 })
    fig.update_traces(marker_size = 4)
    st.plotly_chart(fig, use_container_width=False, )

    st.markdown("On voit que les données sont très éparpillées malgré nos efforts !")
    
    st.markdown("""# RESULTAT: Input shape (8478, 96)""")
    
    
    