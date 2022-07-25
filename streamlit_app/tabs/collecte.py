import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

title = "Web Scraping"
sidebar_name = "Web Scraping"





def run():

    st.title(title)
    st.image('../streamlit_app/assets/scrapping/featured_image-6.webp')

    st.markdown(
        """
    Notre dataset contient une liste de jeux vidéo qui ont vendu plus de 100,000 copies,elle est généré par le scrapping du site vgchartz.com.

    Les variables collectées sont  : 

    \t- Rank - Classement
    \t- Name - Nom du jeux
    \t- Platform - Platforme de jeux (i.e. PC,PS4, etc.)
    \t- Year - année de sortie du jeux
    \t- Genre - Genre du jeux
    \t- Publisher - Publisher of the game
    \t- NA_Sales - Ventes en Amérique du Nord (en millions)
    \t- EU_Sales - Ventes en Europe (en millions)
    \t- JP_Sales - Ventes au  Japan (en millions)
    \t- Other_Sales - Ventes dans le reset du monde (en millions)
    \t- Global_Sales - Total des ventes. 

    Le code qui a permis de collecter ces données est disponible sur Github https://github.com/GregorUT/vgchartzScrape.
    le script utilise la librairie BeautifulSoup de Python pour collecter les données
    la dataset contient  16,598 lignes. 2 lignes ont été  supprimées à cause de manque d'informations
        """
    )


    st.markdown(
        """
        ## Limitations de Vgsales.csv
    Le but inital de notre projet est de prédire les ventes des jeux , cependant, le nombre et la qualité des variables 
    qualitatives et l'absence des variables numérique ne permet pas de construire un modéle qui peut prédire les ventes.

    Nous étaient donc contrains à rechercher plus d'informations dans le web , pour enrichier notre base de données
    Nous avons trouvé plusieurs sites, dans lequels ont peux extraire des données sur les jeux vidéo 
       # Scrapping
    Le tableu ci dessous décrit les différents sites visités et utilisés pour le Web Scrapping ainsi que les variables collectées
    

       """
    )

    df = pd.read_csv('../streamlit_app/assets/scrapping/tableau_scrapping.csv')
    st.table(df)
    st.markdown(
        """
    Pour le scrapping des données, nous avons utilisées plusieurs méthodes et outils :

     -  Scripting Python
     -  librairies : Beatutiful Soup , Selenium , Pandas ...
     -  Extensions navigateur: Selector Gadget
     -  Api : Youtube, Twitch, Twitter ...

      """
    )
    
     
    st.markdown(
            """
            ## Wikipedia:
            Wikipédia est une encyclopédie universelle et multilingue créée par Jimmy Wales et Larry Sanger le 15 janvier 2001. 
            Il s'agit d'une œuvre libre, c'est-à-dire que chacun est libre de la rediffuser. Gérée en wiki dans le site web wikipedia.org grâce au logiciel MediaWiki, 
            elle permet à tous les internautes d'écrire et de modifier des articles, ce qui lui vaut d'être qualifiée d'encyclopédie participative. 
            Elle est devenue en quelques années l'encyclopédie la plus fournie et la plus consultée au monde.

            Chaque jeux video a un article dédié sur Wikipedia, dans lequel on trouve plusieurs informations qui peuvent 
            nous aider à enrichir notre Dataset 
            """
        )
 
    st.image('../streamlit_app/assets/scrapping/GTAV.png')


    st.markdown(
            """
            On va utiliser les données de l'infobox qui de Wikipedia qui résume plusieurs informations qui 
            peuvent nous étre utiles et enrichir notre Dataset , comme Series pour la franchise , le Mode de jeux (multijouer ..)

            on a cherché aussi dans les sections des articles si le jeux a recu des prix ou non 

            les colonnes qu'on a scrappé sont les suivantes:

            """
        )
    df2 = pd.read_csv('../streamlit_app/assets/scrapping/wikipedia_sample.csv',sep = ';')
    st.table(df2)
    


    st.markdown(
            """
            ## Youtube:
            Dans Youtube, il y a plusieurs vidéo qui parlent des jeux vidéo, nous avons donc choisir
            la méme chaine pour scrapper les statistiques de vues et de commentaire , pour avoir des résultats
            cohérentes, nous avons choisir aussi l'une des chaines les plus connues pour les critiques des jeux vidéo
            la chaine IGN, du fameux site web IGN

            nous avons utilisé l'API Youtube pour le scrapping


            """
        )
 
    st.image('../streamlit_app/assets/scrapping/Youtube.png')

    st.markdown(
            """
            
            les colonnes collectées sont les suivantes

            
            """
        )

    df3= pd.read_csv('../streamlit_app/assets/scrapping/Youtubesample.csv',sep = ';')
    st.table(df3)

    st.markdown(
            """

            ## Metacritics:

            Metacritic est un site Web américain de langue anglaise qui collecte les notes attribuées aux albums de musique, jeux vidéo, films, émissions de télévision,
            DVD et livres dans les tests anglophones. Pour chaque produit, un score numérique est obtenu et le total est ramené à une moyenne en pourcentage.
            Metacritic a été fondé en 1999 et racheté en août 2005 par la société CNET Networks, elle-même rachetée par CBS Corporation en 2008.
            
            Nous avons collectés les scores de metacritics, dont on a deux types de scores : 
            les scores des utilisateurs du site, et les scores des professionnels de critiques de jeux videos


            """
        )
    st.image('../streamlit_app/assets/scrapping/meta.png')

    df3= pd.read_csv('../streamlit_app/assets/scrapping/metasample.csv',sep = ';')
    st.table(df3)
