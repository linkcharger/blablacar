import json
import os
import random
import time
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

API_KEY = 'Ai2mMIEvgVmzWecJlSGIPIAUkhO1nYFM'
API_KEY2 = 'YUDeUUu3E2frGD0xOurnTpFal3lBg2G6'
API_KEY3 = 'E5wFr32IQw6WQTXTsXW3YbFP17o6kW3m'
API_KEY4 = 'pcwU8lsz8UFwWW4stLGh3sgLWal4o4Je'
API_KEY5 = 'JfrsG82WpH4f1kGsLzr4Zg9QdK0Nvvur'
API_KEY6 = 'WDb29izmHCZ1A15b6XNipVuzQHUtd51L'
API_KEY7 = 'TmZUSo3yHRFbGZXpMWvWQl6tLc931rFx'
API_KEY8 = 'eK8x24RxLZ7trItBNCwjMGRyw37yMXsl'
API_KEY9 = '4dJBAHao7BqNcnKHbAmI7ps8UvI5jayC'
API_KEY10 = 'Z5Sua7d2z3iGem394czAnw5Re8KSt3LF'
API_KEY11 = 'p7BFvHeVbz7xXkCzSXnMoWh2l9iqGxIk'
API_KEY12 = 'ju7OYV5XUWuaGbmNgfwJjg7FRVrB8w1h'
API_KEY13 = 'JtSIv9sKtfNRdWw4TMDXOj11NZYj8tnE'
API_KEY14 = 'yoSbiIUNjtUk2w4t1C15AydBMz74AmOq'
API_KEY15 = 'mPZYLAI5KvG67PUAzHas5HhSCFqox541'
API_KEY16 = 'axhDuG6KL9HTDgopB0dP5HRs1Y9VBqbu'


API_DICT = {
    'main': API_KEY,
    'aux1': API_KEY2,
    'aux2': API_KEY3,
    'aux3': API_KEY4,
    'aux4': API_KEY5,
    'aux6': API_KEY6,
    'aux7': API_KEY7,
    'aux8': API_KEY8,
    'aux9': API_KEY9,
    'aux10': API_KEY10,
    'aux11': API_KEY11,
    'aux12': API_KEY12,
    'aux13': API_KEY13,
    'aux14': API_KEY14,
    'aux15': API_KEY15,
    'aux16': API_KEY16,
}

#%%
def getRow(trip):
    '''
    Flattens a BlaBlaCar trip object into a plain list.
    '''
    return {
        'url': trip['links']['_front'],
        'from_lat': trip['departure_place']['latitude'],
        'from_lon': trip['departure_place']['longitude'],
        'to_lat': trip['arrival_place']['latitude'],
        'to_lon': trip['arrival_place']['longitude'],
        'car_comfort': trip['car']['comfort_nb_star'] if 'car' in trip else '',
        'car_maker': trip['car']['make'] if 'car' in trip else '',
        'dep_date': trip['departure_date'],
        'distance': trip['distance']['value'],
        'duration': trip['duration']['value'],
        'price': trip['price_with_commission']['value'],
        'seats': trip['seats']
    }

def rotate(d):
    '''
    Function to rotate dictionary values. Takes a dictionary, returns
    another with values displaced one key to the right.
    '''
    keys = d.keys()
    values = list(d.values())
    values = values[1:] + values[:1]
    d = dict(zip(keys, values))
    return d

def getTrips(origin, startdate, dataset, log_dest):
    '''
    Iterates over the BlaBlaCar trip endpoints.
    Checks for multiple page results, rotates keys and returns
    indexed results by department.

    - origin: takes city row
    - startdate: takes a datetime
    - dataset: Frame containing all possible destinations
    - log_dest: Takes a path to dump json results
    
    Returns
    -------
    :trips:

    '''
    trips = []

    # local_dict = API_DICT
    
    iterator = tqdm(dataset[~(dataset.index == origin.index[0])].iterrows())
    
    for i, row in iterator:
        iterator.set_description(f'{origin.Commune}')
        
        # local_dict = rotate(local_dict)
        KEY = random.choice(list(API_DICT.values()))
        page = None
        iterr_list = []

        URL = "https://public-api.blablacar.com/api/v3/trips"
        CUR = "EUR"

        HEADERS = {
            'Content-Type': "application/json",
            'Cache-Control': "no-cache"
        }
        QS_BASE = {
            "key": KEY,
            "currency": CUR,
            "from_coordinate": origin['coord'],
            "to_coordinate": row['coord'],
            "start_local_date": startdate
        }

        querystring = dict(
            **QS_BASE
        )

        url = "https://wtfismyip.com"
        proxy_host = "gate.smartproxy.com"
        proxy_port = "7000"
        proxy_user = "blablacar"
        proxy_password = "blablacar_pass"

        # proxies = {
        #     "https": f"http://user-{proxy_user}:{proxy_password}@{proxy_host}:{proxy_port}/",
        #     "http": f"http://user-{proxy_user}:{proxy_password}@{proxy_host}:{proxy_port}/",
        # }

        rj = None
        
        while rj is None:
            try:
                response = requests.request(
                    "GET",
                    URL,
                    headers=HEADERS,
                    params=querystring,
                    # proxies=proxies
                )
    
                rj = response.json()
    
                iterr_list.extend(rj['trips'])

                time.sleep(random.uniform(0.5,1))
    
                with open(log_dest, 'a') as f:
                    f.write(json.dumps(rj))
    
                while 'next_cursor' in rj:
    
                    time.sleep(random.uniform(0.5,1))
    
                    page = rj['next_cursor']
    
                    querystring = dict(
                        {'from_cursor': page},
                        **QS_BASE
                    )
    
                    response = requests.request(
                        "GET",
                        URL,
                        headers=HEADERS,
                        params=querystring, 
                        timeout=30,
                        # proxies=proxies
                    )
    
                    rj = response.json()
    
                    iterr_list.extend(rj['trips'])
    
                    with open(log_dest, 'a') as f:
                        f.write(json.dumps(rj))
    
                trips.append(
                    tuple([i, datetime.fromtimestamp(time.time()), iterr_list])
                )

                time.sleep(
                    random.uniform(1, 2)
                )
    
            except ValueError:  # includes simplejson.decoder.JSONDecodeError
                print(f'Decoding JSON has failed for trips from {row.Commune}')
                with open(log_dest, 'a') as f:
                    f.write('ValueError')
                    
                trips.append(tuple([i, datetime.fromtimestamp(time.time()), None]))
    
            except KeyError as e:
                remaining_calls = response.headers['x-ratelimit-remaining-day']
                print(e, f'with KEY {KEY}. Remaining calls: {remaining_calls}')
                if remaining_calls != 0:  
                    time.sleep(15)
                if remaining_calls == 0:
                    time.sleep(15)
                    KEY = local_dict['aux1']
                    QS_BASE['key'] = KEY
                continue
                
            except ConnectionError as e:
                print(e)
                pass

    return trips

def uniquifier(path):
    filename, extension = os.path.splitext(path)
    counter = 0
    path = filename + "_" + str(counter) + extension

    while os.path.exists(path):
        counter += 1
        path = filename + "_" + str(counter) + extension

    return path, counter


"""
API_requests.
--------------
Maps the API functions on ori-dest pairs, processes data and stores it.

@author: David
"""



today = date.today()

#%% Paths
bbcardir = Path(os.environ["BLABLACAR"])
scriptsdir = bbcardir / 'git_scripts'
datadir = bbcardir / 'data'
outdir = datadir / 'scraper' / 'output'

os.chdir(scriptsdir / 'scraper')


#%% Log and output files
log_dump = datadir / 'scraper' / '_API_dumps' / f'{today}_JSON.txt'
file_to_operate, day_counter = (
    uniquifier(
        str(datadir / 'scraper' / '_API_dumps' / 'csv' / f'{today}_trips.csv')
    )
)

#%% Process coord combination df
coordinate_mapper = pd.read_csv(
    datadir / 'scraper' / 'misc' / 'hotels-de-prefectures-fr.csv',
    sep=';'
)

coordinate_mapper = (
    coordinate_mapper.loc[
        (pd.to_numeric(coordinate_mapper['DeptNum'], errors='coerce').notnull()) &
        (pd.to_numeric(coordinate_mapper['DeptNum'], errors='coerce') < 100)
    ]
    .assign(
        Commune=lambda df:
            df['Commune'].str.normalize('NFKD')
            .str.encode('ascii', errors='ignore')
            .str.decode('utf-8'),
        coord=lambda df: df['LatDD'].round(4).astype(str) + ',' + df['LonDD'].round(4).astype(str),
        DeptNum=lambda df: df['DeptNum'].astype(int)
    )
    .set_index('DeptNum')
)
#%% Call in cities to request trips for
majors_df = (
    coordinate_mapper
    .loc[
        (coordinate_mapper.Commune == 'Paris') |
        (coordinate_mapper.Commune == 'Marseille') |
        (coordinate_mapper.Commune == 'Lyon') |
        (coordinate_mapper.Commune == 'Toulouse') |
        # (coordinate_mapper.Commune == 'Nimes') |
        (coordinate_mapper.Commune == 'Nice') |
        (coordinate_mapper.Commune == 'Rennes') |
        # (coordinate_mapper.Commune == 'Nantes') |
        (coordinate_mapper.Commune == 'Lille') |
        # (coordinate_mapper.Commune == 'Saint-Etienne') |
        (coordinate_mapper.Commune == 'Bordeaux') |
        (coordinate_mapper.Commune == 'Strasbourg') |
        (coordinate_mapper.Commune == 'Limoges')
    ]
    .copy()
)
# coordinate_mapper = coordinate_mapper.loc[coordinate_mapper.index==69]
#%% API calls
local_df = majors_df.copy()

trips_list = []

while not local_df.empty:
    for i, row in local_df.iterrows():
        try:
            results = getTrips(
                origin=row,
                startdate=today,
                dataset=coordinate_mapper,
                log_dest=log_dump
            )
            
            trips_list.append([i, results])
            local_df.drop(i, inplace=True)
        
        # Key error includes empty local_df. Break outside while loop
        except KeyError as e:
            print(e)
            break
        
        # If any other error, continues iterrows 
        except Exception as e:
            print(e)
            pass
    cur_length = local_df.shape[0]
    scraped_length = len(trips_list)
    
    print(f"TRIP LOOP COMPLETED: Retry {cur_length} trips. {scraped_length} trips have been scraped.")









########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################








API_results = pd.DataFrame(trips_list, columns=['DeptNum', 'results'])
API_results.set_index('DeptNum', inplace=True)

#%% Data Wrangling
majors_df = majors_df.merge(
    API_results,
    left_index=True,
    right_index=True,
    how='left'
)

# Split trips to different departments
results = (
    majors_df
    .copy()
    .explode('results')
)

# Split destination and trip information
results[['destination', 'API_scrape_time', 'trips']] = results['results'].apply(pd.Series)

# Explode json list data for individual blablacars
results = (
    results
    .drop(columns=['results'], axis=1)
    .explode('trips')
    .assign(trips=lambda df: df.trips.fillna({i: {} for i in results.index}))
    .reset_index()
)

# Flatten json individual trip data
results = results.join(pd.json_normalize(results['trips']))
results['trip_id'] = results.link.str.extract('id=(.*)')
results['waypoints'] = results.waypoints.fillna({i: [{}, {}] for i in results.index})

# Split and flatten start and endpoint information
results['start'] = [x[0] for x in results['waypoints']]
results['end'] = [x[1] for x in results['waypoints']]

results = results.join(pd.json_normalize(results['start'].tolist()).add_prefix("start."))
results = results.join(pd.json_normalize(results['end'].tolist()).add_prefix("end."))

# Extract actual trip identifier (numeric)
results['num_id'] = results.trip_id.str.extract('(\d+)')

results.drop(
    columns=[
        'Nom',
        'DeptNom',
        'LatDD',
        'LonDD',
        'trips',
        'waypoints',
        'start',
        'end'
    ],
    axis=1,
    inplace=True
)

results['day_counter'] = day_counter

results = results.sample(frac=1) # shuffle

#%%
results.to_csv(file_to_operate)
