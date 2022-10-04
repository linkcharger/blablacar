# %%
import json
from glob import glob

import pandas as pd
import numpy as np
import plotly

from tqdm import tqdm
tqdm.pandas()

pd.set_option('display.max_columns', 250)
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_colwidth', None)
pd.options.display.float_format = '{:.2f}'.format
plotly.offline.init_notebook_mode(connected=True)

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

def ethnicity_parse(row, role: str):
    '''
    For each driver, compute the average share of passengers they accepted with xxx ethnicity.
    What changes is the denominator:
        - for '_total' variables, the denominator is the number of people in the driver's ratings (eg. 10% of the ppl that reviewed them (from rides past) are black)
        - for '_total_obs' variables, the denominator is the number of people we observe having been passengers of this driver (eg. 5% or 20% of the people we directly observe are black)

    We have to remember that when we scrape, we are essentially taking a subset of all rides ever taken. 
    Therefore, the ethnicities based on this subset might in some way deviate (stochastically or structurally) from the complete set of rides. 
    Thus, if nothing changed structurally (before and during scraping), we would expect the ratings based on the complete set to be less biased and more precise.
    '''

    reviewsThisTrip = pd.json_normalize(row['ratings'])                                                                 # get reviews of this trip

    try:
        reviewsThisTrip = reviewsThisTrip.loc[lambda df: df.sender_uuid != '']                                          # only get those with an ID
        reviewsThisTrip.dropna(subset=['sender_uuid'], inplace=True)
    except:
        pass


    if reviewsThisTrip.shape[0] > 0 and 'role' in reviewsThisTrip:

        reviewersThisTripThisRole = reviewsThisTrip[reviewsThisTrip['role'] == role]['sender_uuid'].tolist()            # get IDs of reviewers being either drivers or passengers

        temp = ratings_ethnicities[ratings_ethnicities['ID'].isin(reviewersThisTripThisRole)].dropna(subset=['ethnicity']) # get subset of reviewers with their ethnicities who were reviewers for this trip

        total = len(reviewersThisTripThisRole) if temp.shape[0] != 0 else 1
        total_obs = temp.shape[0]



        asian   = temp.ethnicity.str.count('Asian').sum()
        black   = temp.ethnicity.str.count('Black').sum()
        indian  = temp.ethnicity.str.count('Indian').sum()
        latino  = temp.ethnicity.str.count('Latino_Hispanic').sum()
        mideast = temp.ethnicity.str.count('Middle Eastern').sum()
        white   = temp.ethnicity.str.count('White').sum()
        norace  = temp.ethnicity.str.count('No Race').sum()

        asian_pc    = temp.ethnicity.str.count('Asian').sum() / total
        black_pc    = temp.ethnicity.str.count('Black').sum() / total
        indian_pc   = temp.ethnicity.str.count('Indian').sum() / total
        latino_pc   = temp.ethnicity.str.count('Latino_Hispanic').sum() / total  
        mideast_pc  = temp.ethnicity.str.count('Middle Eastern').sum() / total
        white_pc    = temp.ethnicity.str.count('White').sum() / total
        norace_pc   = (temp.ethnicity.str.count('No Race').sum() + total - total_obs) / total




    elif reviewsThisTrip.shape[0] == 0 or 'role' not in reviewsThisTrip:
        total, total_obs, asian, black, indian, latino, mideast, norace, white = 0, 0, 0, 0, 0, 0, 0, 0, 0
        asian_pc, black_pc, indian_pc, latino_pc, mideast_pc, norace_pc, white_pc = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan




    return (
        total, 
        total_obs, 
        asian, 
        asian_pc, 
        black, 
        black_pc, 
        indian, 
        indian_pc, 
        latino, 
        latino_pc, 
        mideast, 
        mideast_pc, 
        norace, 
        norace_pc, 
        white, 
        white_pc
        )



COLUMNS = {
    'passenger': [
        'pass_total', 
        'pass_total_obs', 
        'pass_asian', 
        'pass_asian_pc', 
        'pass_black', 
        'pass_black_pc', 
        'pass_indian', 
        'pass_indian_pc', 
        'pass_latino', 
        'pass_latino_pc', 
        'pass_mideast', 
        'pass_mideast_pc', 
        'pass_norace', 
        'pass_norace_pc', 
        'pass_white', 
        'pass_white_pc'
        ],

    'driver' : [
        'driv_total', 
        'driv_total_obs', 
        'driv_asian', 
        'driv_asian_pc', 
        'driv_black', 
        'driv_black_pc', 
        'driv_indian', 
        'driv_indian_pc', 
        'driv_latino', 
        'driv_latino_pc', 
        'driv_mideast', 
        'driv_mideast_pc', 
        'driv_norace', 
        'driv_norace_pc', 
        'driv_white', 
        'driv_white_pc'
        ]
}






# %%
predictions = pd.read_csv(f'../05_deepface/data/ethnicity_predictions_2022-06-01.csv')
alreadyProcessedDates = [path[-14:-4] for path in glob('data/*.csv')]
alreadyProcessedDates



# %%
for file in glob('../03_scrape_trip_details/01_data-raw_JSON_trip_details/*trips.txt'):
    date = file[-20:-10]
    day_file = []
    
    # skip if already processed
    if date in alreadyProcessedDates: continue


    for role in ['passenger', 'driver']:

        scrape_df = parseJSONtoDF(file)                                                                                         # get raw detailed data about trip
        scrape_df['file_wbs'] = date
        scrape_df['num_id'] = scrape_df.trip_id.str.extract('(\d*)-').astype('int64')

        ratingsSubset = scrape_df[['num_id', 'driver_id', 'ratings']]                                                           # take column subset
        
        reviewerDF = ratingsSubset.drop_duplicates(subset=['driver_id'], keep='last', inplace=False)                            # keep last observation of driver (with most ratings)
        reviewerDF = reviewerDF.explode('ratings')      
        reviewerDF = reviewerDF.dropna(subset=['ratings'])
        reviewerDF = pd.json_normalize(reviewerDF.ratings)[['sender_uuid', 'sender_display_name', 'sender_profil_picture']]     # extract json info
        reviewerDF = reviewerDF.drop_duplicates(subset=['sender_uuid'], keep='last')
        reviewerDF = reviewerDF.rename(columns={'sender_uuid': 'ID', 'sender_profil_picture': 'thumbnail'})

        ratings_ethnicities = pd.merge(reviewerDF, predictions, on='ID', how='left', indicator=False)                           # is used in ethnicity_parse() below

        trips = ratingsSubset.drop_duplicates('num_id')                                                                         # setup main output table
        
        trips[COLUMNS[role]] = trips.progress_apply(lambda row: ethnicity_parse(row, role=role), axis=1, result_type='expand')  # main numerical operation: calculate shares
       
        trips = trips.drop('ratings', axis=1)
        day_file.append(trips)

    trips = pd.merge(day_file[0], day_file[1], on=['num_id', 'driver_id'])
    trips.to_csv(f'data/ethnicity_trips_{date}.csv')
        