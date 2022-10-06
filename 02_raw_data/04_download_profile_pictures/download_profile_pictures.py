# %%
import os, re, requests, datetime
from typing import Union
from pathlib import Path
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
    import json

    with open(file) as f:
        data = json.loads(f.read())
           
    data = {k: v for x in data for k in x.keys() for v in x.values() if v['status']}
           
    trip_df = pd.DataFrame.from_dict(data, orient='index')
    
    t_list = []
    for i, row in trip_df.iterrows():
        # print(i)
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
    
    # print('made it')
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

class PFPdownloader:

    def __init__(self, file, userTypes: Union[Path, list], outputPath) -> None:
        '''
        Parses JSON file and downloads profile pictures (PFPs) for 
        drivers, passengers or/and reviewers with a normal GET request.
        
        Parameters
        ----------
        file: JSON file with raw data for the day (detailed trip information)
        userTypes: the types of blablacar users for whom you want to download profile pictures (pfps)
        outputPath: the folder in which to store all pictures
        '''
        assert set(userTypes) <= set(['drivers', 'passengers', 'reviewers']), ('Unknown user type string used. Please choose between drivers, passengers, reviewers.')
        if not isinstance(userTypes, list): userTypes = [userTypes]
        self.userTypes = userTypes
        self.outputPath = outputPath
        self.types = {
            'drivers': {
                'func': self.prepareDriverData, 
                'data': [],
                },
            'passengers': {
                'func': self.preparePassengerData, 
                'data': [],
                },
            'reviewers': {
                'func': self.prepareReviewerData, 
                'data': [],
                }
            }

        self.data = parseJSONtoDF(file)
        self.data['file_date'] = re.search(('\d{4}-\d{2}-\d{2}'), file).group(0), 
        self.data['num_id'] = self.data.trip_id.str.extract('(\d*)-').astype('int64')

    def prepareDriverData(self):
        '''
        prepares driver dataset for downloading in the next step. 
        
        - select relevant columns from the DF of the day (but also irrelevant ones like gender and name, when all were gonna use is the ID and URL?)
        - drop duplicates
        - rename to uniform name scheme (so that download function knows how to deal with it)
        '''
        driverData = (
            self.data[['num_id', 'driver_id', 'driver_display_name', 'driver_gender', 'driver_thumbnail']]
            .drop_duplicates(subset=['driver_id', 'driver_thumbnail'])
            .rename(columns={
                'driver_id': 'ID', 
                'driver_thumbnail': 'thumbnail'
                })
        )
        self.types['drivers']['data'] = driverData
    
    def preparePassengerData(self):
        '''
        prepares passenger dataset for downloading in the next step. 

        - select relevant columns from the DF of the day 
        - explode to get more info from JSON
        - rename to uniform name scheme
        '''
        passengerData = self.data[['num_id', 'passengers']]             # start with DF of the day
        passengerData = passengerData.explode('passengers')             # select only passenger column and explode it
        passengerData.dropna(subset=['passengers'], inplace=True)       

        JSONpassengers = pd.json_normalize(passengerData.passengers)    # get more details from JSON part   
        passengerData.reset_index(inplace=True, drop=True)

        passengerData = pd.merge(                                       # merge the previous passenger information (only the num_id i guess?) to the more detailed info from JSON (whats the index that its being merged on?)
            passengerData[['num_id']], 
            JSONpassengers[['id', 'display_name', 'gender', 'thumbnail']],
            left_index=True,
            right_index=True
            )

        passengerData.rename(columns={'id': 'ID'}, inplace=True)
        self.types['passengers']['data'] = passengerData

    def prepareReviewerData(self):
        '''
        prepares reviewer dataset for downloading in the next step. 

        - select relevant columns from the DF of the day 
        - explode to get more info from JSON
        - rename to uniform name scheme
        '''
        reviewerData = self.data[['num_id', 'driver_id', 'ratings']]
        reviewerData = reviewerData.explode('ratings')
        reviewerData.dropna(subset=['ratings'], inplace=True)
        reviewerData.drop_duplicates(subset=['driver_id'], keep='last', inplace=True)

        JSONreviewers = pd.json_normalize(reviewerData.ratings)

        reviewerData = JSONreviewers[['sender_uuid', 'sender_display_name', 'sender_profil_picture']]
        reviewerData.drop_duplicates(subset=['sender_uuid'], keep='last', inplace=True)
        reviewerData.rename(columns={
            'sender_uuid': 'ID', 
            'sender_profil_picture': 'thumbnail'
            }, inplace=True)

        self.types['ratings']['data'] = reviewerData
    
    def setupDownloads(self):
        '''
        preparations for downloading pictures for one or several userTypes. 
        - for each userTypes, prepare a dataframe of relevant people (with URLs and IDs)
        - exclude those already downloaded
        - for the rest, download the pictures
        '''
        for file in self.files:
            existingPFPs = [t[:-5] for t in os.listdir('01_data-thumbnails_to_label') if 'jpeg' in t] ## this should probably be refactored out of here, to not have to read 500k files every time. or make it a list in a file.
            
            for type in self.userTypes:
                try:
                    self.types[type]['func']()                                                      # prepare respective data: if userTypes is 'drivers', launch self.prepareDriverData()
                    dataToProcess = self.types[type]['data']                                        # retrieve data prepared by driver() function
                    dataToProcess = dataToProcess.loc[~dataToProcess['ID'].isin(existingPFPs)]      # exclude existing pfps
                    
                    dataToProcess.progress_apply(lambda row: self.downloadPicture(row), axis=1)     # pass row to it (to get both URL and ID), download picture from URL and name it ID

                except KeyError:
                    print(f'No new thumbnails for day {str(file.__name__)}')
                    continue

    def downloadPicture(self, row):
        '''
        download picture from URL and name it ID
        '''
        try:
            response = requests.get(row['thumbnail'])

            # if request successful, save it to
            if response.status_code == 200:
                with open(f'{self.outputPath}/{row["ID"]}.jpeg', 'wb') as f:
                    f.write(response.content)
        except:
            pass












# %% ################################################################################################################################################################
for file in glob('../03_scrape_trip_details/01_data-raw_JSON_trip_details/*trips.txt'):
    
    instance = PFPdownloader(
        file=file,        
        userTypes=['reviewers'],
        outputPath='01_data-profile_pictures',
        )

    instance.setupDownloads()





