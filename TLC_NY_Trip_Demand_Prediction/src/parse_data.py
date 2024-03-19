import requests
from bs4 import BeautifulSoup
import fake_useragent 
import numpy as np
import pandas as pd 

from tqdm import tqdm
import os 

def get_page_data(url):
    """
    Gets HTML page using requests, fake_useragent and BeautifulSoup libraries 

    Arguments:
    url: str
        URL of the main page with csv links 
    Returns: 
    html: lxml
    """
    user = fake_useragent.UserAgent().random
    header = {'user-agent': user}
    response = requests.post(url, headers = header)
    return BeautifulSoup(response.text, 'lxml')

def download_data(year_start, year_end, url, folder_name):
    """
    Downloads links for csv files. Each link is passed into pandas and read
    Finally, pandas saves it as csv into a provided directory
    
    Arguments: 
    year_start: int 
        Starting year
    year_end: int 
        Ending year
    url: str 
        URL of the main page with csv links
    folder_name: str
        Result folder name with parsed data 
    Returns:
    --------
    None 
        Downloads files and save them into a provided directory using pandas
    """
    data_links = []
    years = np.arange(year_start, year_end + 1, 1)
    os.mkdir(folder_name)
    
    html = get_page_data(url)
    for year in years:
        # Extract links 
        for monthly_data in html.select_one(f'#faq{year}').find_all('ul'):
            data_links.append(monthly_data.find_all('a', href=True)[0]['href'])      
    # Download a file 
    for link in tqdm(data_links):
        current_file_name = link.split('/')[-1]
        print(f'Downloading: {current_file_name}')
        file_path = os.path.join(os.getcwd(), 'taxi_data', current_file_name)  
        df = pd.read_csv(link)
        df.to_csv(file_path, index=False)
        print(f'File {current_file_name} Successfully Saved!')


# Main parameters 
URL = 'https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page'
START_YR = 2019
END_YR = 2021
FOLDER_NAME = 'taxi_data'

# Script run 
download_data(year_start=START_YR, year_end=END_YR, url=URL, folder_name=FOLDER_NAME)