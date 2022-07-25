import pandas as pd
from datetime import datetime

def add_game_key(df,col='Name'):
    """ add normalized 'game_key' for col (default named 'Name')"""
    df['game_key'] = df['Name'].str.normalize('NFKD').str.encode('ascii',errors='ignore').str.decode('utf-8').str.replace('[^A-Za-z0-9]+', '-', regex=True).str.replace('^-+|-+$', '', regex=True).str.lower()
    
    return df
    
def convert_date_metacritic(df,col_date):
    """Add day month year and release_date columns
    parameters:
        df: metacritic df
        col_date: the column date containing the date in string format
                  mmm dd, yyyy (example Apr 30, 1995)
    return
        df: initial df with additional columns"""
    
    df[['month_str', 'day', 'year']] = df[col_date].str.extract(r'(\w+) +(\d+), +(\d+)')
    df['month_str'] = df['month_str'].fillna(df['month_str'].mode()[0])
    df['month'] = df['month_str'].apply(lambda x: datetime.strptime(x, "%b").month if (type(x) == str) else x)
    df['release_date']= pd.to_datetime(df[['year','month', 'day']])
    df = df.drop(columns=[col_date,'month_str'])
    return df
    
    

