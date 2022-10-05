# %%
from glob import glob

import pandas as pd
pd.set_option('display.max_columns', 250)
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_colwidth', None)
pd.options.display.float_format = '{:.2f}'.format

import plotly
plotly.offline.init_notebook_mode(connected=True)

from tqdm import tqdm
tqdm.pandas()

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



#%%######################## load all raw trip details data into one df per usertype ###############################
driverData = []
passengerData = []
reviewerData = []

for file in tqdm(glob('../03_scrape_trip_details/01_data-raw_JSON_trip_details/*trips.txt')[:2]):
    date = file[-20:-10]
    
    scrape_df = parseJSONtoDF(file)
    scrape_df['file_wbs'] = date
    scrape_df['num_id'] = scrape_df.trip_id.str.extract('(\d*)-').astype('int64')

    driverData.append(scrape_df[['num_id', 'driver_id', 'driver_display_name', 'driver_gender', 'driver_thumbnail']])
    passengerData.append(scrape_df[['num_id', 'passengers']])
    reviewerData.append(scrape_df[['num_id', 'driver_id', 'ratings']])

driverDF =  pd.concat(driverData)
passengerDF =  pd.concat(passengerData)
reviewerDF =  pd.concat(reviewerData)




#%%######################## clean and save driver data ###############################
driverDF = driverDF.drop_duplicates(subset=['driver_id', 'driver_thumbnail'])                           # keeps first by default
driverDF = driverDF.rename(columns={
    'driver_id': 'ID', 
    'driver_thumbnail': 'thumbnail', 
    })
driverDF.to_csv('data/01_drivers_only.csv')
driverDF





#%%######################## clean and save passenger data ###############################
passengerDF = passengerDF.explode('passengers')
passengerDF = passengerDF.dropna(subset=['passengers'])
passengerDF = passengerDF.reset_index(drop=True)

json_passengers = pd.json_normalize(passengerDF.passengers)

passengerDF = pd.merge(
    passengerDF[['num_id']], 
    json_passengers[['id', 'display_name', 'gender', 'thumbnail']],
    left_index=True,
    right_index=True
    )

passengerDF = passengerDF.rename(columns={'id':'ID'})
passengerDF.to_csv('data/01_passengers_only.csv')
passengerDF




#%%######################## clean and save reviewer data ###############################
reviewerDF = reviewerDF.drop_duplicates(subset=['driver_id'], keep='last')
reviewerDF = reviewerDF.explode('ratings')
reviewerDF = reviewerDF.dropna(subset=['ratings'])

json_ratings = pd.json_normalize(reviewerDF.ratings)

reviewerDF = json_ratings[['sender_uuid', 'sender_display_name', 'sender_profil_picture']]

reviewerDF = reviewerDF.drop_duplicates(subset=['sender_uuid'], keep='last')
reviewerDF = reviewerDF.rename(columns={'sender_uuid': 'ID', 'sender_profil_picture': 'thumbnail'})
reviewerDF.to_csv('data/01_reviewers_only.csv')     # formerly 'ratings.csv'
reviewerDF














#%%####################### merge ethnicities to each ############################
predictions = pd.read_csv(f'../05_deepface/data/ethnicity_predictions_2022-06-01.csv')
predictions = (
    predictions[['image', 'race 1', 'accuracy race prediction 1']]
    .rename(columns={
        'image': 'ID', 
        'race 1': 'ethnicity',
        'accuracy race prediction 1': 'ethnic_acc', 
        })
    .assign(
        ID = lambda df: df['ID'].str[:-5]
        )
    )
predictions.to_csv('data/ethnicity_predictions.csv')



# ethnicity + drivers dataframe
drivers_eth = pd.merge(driverDF, predictions, on='ID', how='left', indicator=False)
drivers_eth.to_csv('data/02_drivers+ethnicity.csv')


# ethnicity + passengers dataframe
passengers_eth = pd.merge(passengerDF, predictions, on='ID', how='left', indicator=False)
passengers_eth.to_csv('data/02_passengers+ethnicity.csv')


# ethnicity + ratings dataframe
ratings_eth = pd.merge(reviewerDF, predictions, on='ID', how='left', indicator=False)
ratings_eth.to_csv('data/02_ratings+ethnicity.csv')


