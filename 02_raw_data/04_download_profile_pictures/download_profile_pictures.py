# %%
import os, re, requests, datetime
from typing import Union
from pathlib import Path
import json
from glob import glob

import pandas as pd
import numpy as np

from dateutil import tz
from tqdm import tqdm
tqdm.pandas()
pd.options.mode.chained_assignment = None
pd.set_option('display.max_columns', 250)
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_colwidth', None)
pd.options.display.float_format = '{:.2f}'.format

def parseJSONtoDF(file):
    '''
    Parses nested dictionaries into a dataframe

    Parameters
    ----------
    file : dumped json list

    Returns
    -------
    output_df : trip dataframe

    '''
    with open(file) as f:
        data = json.loads(f.read())
           
    data = {k: v for x in data for k in x.keys() for v in x.values() if v['status']}
           
    trip_df = pd.DataFrame.from_dict(data, orient='index')
    
    t_list = []
    for i, row in trip_df.iterrows():
        print(i)
        try:
            rating_info = [item for sublist in row['rating'] for item in sublist]
            
            ride_df = pd.DataFrame([row['ride']], columns=row['ride'].keys())
            
            ride_df['ratings'] = [rating_info]
            ride_df['web_scrape_time'] = row['web_scrape_time']
            
            ride_df = (
                ride_df
                .join(pd.json_normalize(ride_df['driver']).add_prefix('driver_'))
                .join(pd.json_normalize(ride_df['multimodal_id']))
                .join(pd.json_normalize(ride_df['vehicle']).add_prefix('vehicle_'))
                .join(pd.json_normalize(ride_df['seats']).add_prefix('seats_'))
            )
        
            t_list.append(ride_df)
                
        except Exception:
            # Skips deleted trips
            pass
        
    output_df = pd.concat(t_list)
    
    print('made it')
    cols = [
        col
        for col in output_df.columns 
        if col.startswith('passengers') 
        or col.startswith('driver_')
        or col.startswith('vehicle_')
        or col.startswith('seats_')
        or col.startswith('ratings')
        or col.startswith('web_scrape')
        ]
    
    output_df = (
        output_df[['id', 'comment', 'flags'] + cols]
        .rename(columns={'id': 'trip_id'})
    )
    
    output_df.reset_index(drop=True, inplace=True)
    
    return output_df

class ThumbnailParser:

    def __init__(self, inputDF, date, userType: Union[Path, list], outputPath) -> None:
        '''
        Class parses any number of pickle files from the parsed output and stores
        thumbnails for drivers, passengers or/and reviewers.
        
        Parameters
        ----------
        files : parsed_output files
        utype : user types
        output : output path
        '''
        assert set(userType) <= set(['drivers', 'passengers', 'ratings']), ('Unknown user type string used. Please choose between drivers, passengers, ratings.')
        if not isinstance(userType, list): userType = [userType]
        self.utype = userType
        self.outputPath = outputPath
        self.types = {
            'drivers': {'func': self.drivers, 'data': []},
            'passengers': {'func': self.passengers, 'data': []},
            'ratings': {'func': self.ratings, 'data': []}
            }

        self.data = inputDF
        self.data['file_date'] = date
        self.data['num_id'] = self.data.trip_id.str.extract('(\d*)-').astype('int64')


    def drivers(self):
        '''
        Creates drivers dataset, normalizes ID label.
        '''
        self.drivers_df = (
            self.data[['num_id', 'driver_id', 'driver_display_name', 'driver_gender', 'driver_thumbnail']]
            .drop_duplicates(subset=['driver_id', 'driver_thumbnail'])
            .rename(columns={'driver_id': 'ID', 'driver_thumbnail': 'thumbnail'})
        )
        self.types['drivers']['data'] = self.drivers_df
    
    def passengers(self):
        '''
        Creates passengers dataset, normalized ID label and explodes JSON.
        '''
        self.passengers_df = self.data[['num_id', 'passengers']]
        self.passengers_df = self.passengers_df.explode('passengers')
        self.passengers_df.dropna(subset=['passengers'], inplace=True)
        self.json_passengers = pd.json_normalize(self.passengers_df.passengers)
        self.passengers_df.reset_index(inplace=True, drop=True)
        self.passengers_df = pd.merge(
            self.passengers_df[['num_id']], 
            self.json_passengers[['id', 'display_name', 'gender', 'thumbnail']],
            left_index=True,
            right_index=True
        )
        self.passengers_df.rename(columns={'id': 'ID'}, inplace=True)
        self.types['passengers']['data'] = self.passengers_df

    def ratings(self):
        '''
        Creates ratings dataset, normalizes ID label and explodes JSON.
        '''
        self.ratings_df = self.data[['num_id', 'driver_id', 'ratings']]
        self.ratings_df = self.ratings_df.explode('ratings')
        self.ratings_df.dropna(subset=['ratings'], inplace=True)
        self.ratings_df.drop_duplicates(subset=['driver_id'], keep='last', inplace=True)
        self.json_ratings = pd.json_normalize(self.ratings_df.ratings)
        self.ratings_df = self.json_ratings[
            ['sender_uuid', 'sender_display_name', 'sender_profil_picture']
        ]
        self.ratings_df.drop_duplicates(subset=['sender_uuid'], keep='last', inplace=True)
        self.ratings_df.rename(columns={'sender_uuid': 'ID', 'sender_profil_picture': 'thumbnail'}, inplace=True)
        self.types['ratings']['data'] = self.ratings_df

        
    def parser(self, row):
        '''
        Requests a thumbnail and stores it in `output`.
        '''
        try:
            response = requests.get(row['thumbnail'])

            if response.status_code == 200:
                with open(str(self.outputPath) + '/' + row['ID'] + '.jpeg', 'wb') as f:
                    f.write(response.content)
        except:
            pass

    def parse_trips(self):
        '''
        Filters old thumbnails and parses new ones.
        '''
        for file in self.files:
            thumbs = [t[:-5] for t in os.listdir(thumb_datadir) if 'jpeg' in t]
            
            for type in self.utype:
                try:
                    self.types[type]['func']()
                    self.outdata = self.types[type]['data']
                    self.outdata = self.outdata.loc[~self.outdata['ID'].isin(thumbs)]
                    self.outdata.progress_apply(lambda row: self.parser(row), axis=1)
                except KeyError:
                    print(f'No new thumbnails for day {str(file.__name__)}')
                    continue


################################################################################################################################################################


# basedir = Path(os.environ['BLABLACAR'])
# datadir = basedir / 'data'
# scrape_datadir = datadir / 'scraper' / 'output'
# thumb_datadir = datadir / 'thumbnails'
# parsed_trips = scrape_datadir / 'parsed_trips'







# %% ################################################################################################################################################################
if __name__ == '__main__':


    for file in glob('../03_scrape_trip_details/01_data-raw_json_dumps/*trips.txt'):

        oneDayTripsDF = parseJSONtoDF(file)        

        

        instance = ThumbnailParser(
            inputDF=oneDayTripsDF,  
            date=re.search(('\d{4}-\d{2}-\d{2}'), file).group(0)       
            userType=['ratings'],
            outputPath='01_data-thumbnails_to_label',
            )


        instance.parse_trips()



#%%
glob('../03_scrape_trip_details/01_data-raw_json_dumps/*trips.txt')