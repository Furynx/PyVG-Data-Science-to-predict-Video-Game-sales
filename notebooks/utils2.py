import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans

from bs4 import BeautifulSoup
import requests
import re
from time import sleep

# Fonction de récupération des informations sur le jeu indiqué sur Metacritic (scraping)
def scrap_meta_info(link):
    req= requests.get(link, headers = {'User-agent': 'Mozilla/5.0'}, timeout=10).text
    sp= BeautifulSoup(req, 'html')
    
    title= sp.find('h1').text
    #print(title)
    platf= sp.select('.platform a')[0].text.strip()
    
    if sp.select('.publisher a') != []: publi= sp.select('.publisher a')[0].text.strip()
    else : publi= 'None'
    
    release= sp.select('.release_data .data')[0].text

    genre= re.sub(r",\s+", ", ", sp.select('.product_genre')[0].text[10:])
    
    if sp.select('.button') != []: dev= sp.select('.button')[0].text.strip()
    else : dev= 'None'
    sleep(5)
        
    return title, platf, release, publi, dev, genre




def lire():
    df = pd.read_csv("../data/vgsales_eda2.csv", sep=",") 
    print ('dataset shape:', df.shape)
    print ('columns:', df.columns)
    df = df[(df.year>2000) & (df.year<2019) ]
    print ('dataset shape ([2001 ; 2018]):', df.shape)

    return df

def affiche_correlation(df):
    cor = df.corr()
    fig, ax = plt.subplots(figsize=(16,12))
    sns.heatmap(cor, annot=True, ax=ax, cmap='viridis'); #viridis, coolwarm

def enlever_date_ref(df):
    df = df.drop(columns=['day', 'month', 'release_date', 'quarter'])
    return df

def supprimer_val_aberrantes(df):
    isof = IsolationForest(contamination=0.02, n_estimators=100)

    if ('anom' in df.columns):
        df = df.drop(columns=['anom'])
    isof.fit(df.select_dtypes(exclude='object'))
    df['anom'] = isof.predict(df.select_dtypes(exclude='object'))
    #df[df['anom']<0].index
    
    df = df.drop(df[df['anom']<0].index)
    df = df.drop(columns=['anom'])
    
    # On ajoute également deux jeux qui sont tout seul dansleur genre !
    #i= df[df['Name']=='Minecraft'].index.to_list()
    #df.loc[i,'Genre'] = 'Action-Adventure'
    #i = df[df['Genre']=='Education'].index.to_list()
    #df.loc[i,'Genre'] = 'Misc'
    
    df = df.drop(df[df['Name']=='Minecraft'].index)
    df = df.drop(df[df['Genre']=='Education'].index)
    
    #df = df.drop(df[df['Genre']=='MMO'].index)
    #df = df.drop(df[df['Genre']=='Party'].index)
    #df = df.drop(df[df['Genre']=='Visual+Novel'].index)
    
    return df

def OEM(r):
    oem = 'Others'
    if r['Platform'] in ['X360',  'XB', 'XOne']:
        oem = "xbox"
    elif r['Platform'] in ['PS', 'PS2', 'PS3', 'PS4', 'PSP', "PSV"]:
        oem = "playstation"
    elif r['Platform'] in ['Wii', 'WiiU', 'N64', 'GC', 'NS', "3DS", 'DS']:
        oem = "nintendo"
    elif r['Platform'] in ['PC']:
        oem = "PC"
    return oem

def ajoute_oem(df):
    df['oem'] = df.apply(OEM,axis=1)
    return df

def ajoute_dummies(df):
    df = df.join(pd.get_dummies(df.oem,drop_first=True))
    
    # On recup les 12 top Publisher
    publisher_list = df.Publisher.value_counts().head(12).index.tolist()
    df['Publisher_top'] = df.apply(lambda r: r['Publisher'] if r['Publisher'] in publisher_list else 'Others', axis=1)
    df = df.join(pd.get_dummies(df.Publisher_top,drop_first=True,prefix='pub'))
    
    # On recup les 12 top Franchises
    franchise_list = df.Franchise_wikipedia.value_counts().head(12).index.tolist()
    df['Franchise_top'] = df.apply(lambda r: r['Franchise_wikipedia'] if r['Franchise_wikipedia'] in franchise_list else 'Others', axis=1)
    df = df.join(pd.get_dummies(df.Franchise_top,drop_first=True,prefix='lic'))
    
    # dummies years and genre
    df = df.join(pd.get_dummies(df.year,prefix='year.release'))
    df = df.join(pd.get_dummies(df.Genre,prefix='genre',drop_first=True))
    
    return df

def normalise_log(df):
    df['Global_Sales.log'] = np.log(df['Global_Sales'] * 1000000 + 1)
    df['N_pro.log'] = np.log(df['N_pro']+1)
    df['N_user.log'] = np.log(df['N_user']+1)
    return df

def affiche_log(df):
    fig, axes = plt.subplots(2,3,figsize=(16,8))
    sns.histplot(data=df, x='Global_Sales',ax=axes[0][0])
    sns.histplot(data=df, x='N_pro',ax=axes[0][1])
    sns.histplot(data=df, x='N_user',ax=axes[0][2])
    sns.histplot(data=df, x='Global_Sales.log',ax=axes[1][0])
    sns.histplot(data=df, x='N_pro.log',ax=axes[1][1])
    sns.histplot(data=df, x='N_user.log',ax=axes[1][2]);  
    

def affiche_pca_cluster(df,c=5):
    col_ = ['Score_pro', 'Score_user',
            'N_pro', 'N_user', 'compound',
            'N_pro.log', 'N_user.log',
            'year','compound',
            'N_pro.log', 'N_user.log',
            'PC', 'nintendo', 'playstation', 'xbox',
           ]  

    df2 = df[col_].copy()
    scaler = StandardScaler()
    scaled_df2 = pd.DataFrame(scaler.fit_transform(df2), columns=df2.columns)

    kmeans = KMeans(n_clusters = c)
    kmeans.fit(scaled_df2)
    labels = kmeans.labels_
    
    pca = PCA (n_components=0.9) #n_components = 3)
    data_2D = pca.fit_transform(scaled_df2)
    #print ("Explained Variance", pca.explained_variance_ratio_)
    
    fig, ax = plt.subplots(figsize=(12,8))
    scatter = ax.scatter(data_2D[:, 0], data_2D[:, 1], c = labels, cmap=plt.cm.Spectral, alpha=0.8)
    ax.set_xlabel(f'PCA 1 ({np.round(pca.explained_variance_ratio_[0],2)}%)')
    ax.set_ylabel(f'PCA 2 ({np.round(pca.explained_variance_ratio_[1],2)}%)')
    ax.set_title("Données projetées sur les 2 axes de PCA")

    legend1 = ax.legend(*scatter.legend_elements(),
                    loc="lower left", title="Classes")
    ax.add_artist(legend1)
    
    plt.show();
    
    df['labels'] = labels
    
    df = df.join(pd.get_dummies(df.labels,prefix='labels'))
    
    return df