#%%
import json
import random
from copy import deepcopy
from datetime import date, datetime
from glob import glob
from time import sleep, time

import pandas as pd
import requests
from tqdm import tqdm

#######################################################################################################################################

APIdict = {
    'main': 'Ai2mMIEvgVmzWecJlSGIPIAUkhO1nYFM',
    'aux1': 'YUDeUUu3E2frGD0xOurnTpFal3lBg2G6',
    'aux2': 'E5wFr32IQw6WQTXTsXW3YbFP17o6kW3m',
    'aux3': 'pcwU8lsz8UFwWW4stLGh3sgLWal4o4Je',
    'aux4': 'JfrsG82WpH4f1kGsLzr4Zg9QdK0Nvvur',
    'aux6': 'WDb29izmHCZ1A15b6XNipVuzQHUtd51L',
    'aux7': 'TmZUSo3yHRFbGZXpMWvWQl6tLc931rFx',
    'aux8': 'eK8x24RxLZ7trItBNCwjMGRyw37yMXsl',
    'aux9': '4dJBAHao7BqNcnKHbAmI7ps8UvI5jayC',
    'aux10': 'Z5Sua7d2z3iGem394czAnw5Re8KSt3LF',
    'aux11': 'p7BFvHeVbz7xXkCzSXnMoWh2l9iqGxIk',
    'aux12': 'ju7OYV5XUWuaGbmNgfwJjg7FRVrB8w1h',
    'aux13': 'JtSIv9sKtfNRdWw4TMDXOj11NZYj8tnE',
    'aux14': 'yoSbiIUNjtUk2w4t1C15AydBMz74AmOq',
    'aux15': 'mPZYLAI5KvG67PUAzHas5HhSCFqox541',
    'aux16': 'axhDuG6KL9HTDgopB0dP5HRs1Y9VBqbu',
    }


today = date.today()


#######################################################################################################################################

def numberLatestScrapingRun():
    todaysFiles = glob(f'01_data-trips_with_duplicates/{date.today()}*.csv')

    try:
        return int(max([path[-5] for path in todaysFiles]))
    except:
        return -1

def getTrips(origin, destinations, startdate, raw_data_location):
    '''
    Iterates over the BlaBlaCar trip endpoints.
    Checks for multiple page results, rotates keys and returns
    indexed results by department.

    - origin: takes city row
    - destinations: frame containing all possible destinations
    - startdate: takes a datetime
    - raw_data_location: path to dump raw data
    
    Returns
    -------
    list of all trips originating from one city, eg PARIS
    
    structured as follows:
    [
        (destination number (eg of nice),       date,   [trip-PARIS-to-nice1, trip-PARIS-to-nice2, trip-PARIS-to-nice3, trip-PARIS-to-nice4, etc]),
        (destination number (eg of marseille),  date,   [trip-PARIS-to-marseille1, trip-PARIS-to-marseille2, trip-PARIS-to-marseille3, trip-PARIS-to-marseille4, etc]),
        (...),
        (...),
    ]
    

    '''
    tripsOfOneOriginToAllDestinations = []

    
    iterator = tqdm(destinations[~(destinations.index == origin.index[0])].iterrows())
    for destinationDepNum, destinationCityRow in iterator:
        iterator.set_description(f'{origin.Commune}')
        

        # set up request
        KEY = random.choice(list(APIdict.values()))
        page = None
        URL = "https://public-api.blablacar.com/api/v3/trips"
        CUR = "EUR"
        HEADERS = {
            'Content-Type': "application/json",
            'Cache-Control': "no-cache"
            }

        defaultParams = {
            "key": KEY,
            "currency": CUR,
            "from_coordinate": origin['coord'],
            "to_coordinate": destinationCityRow['coord'],
            "start_local_date": startdate
            }
        




        listOfTripDictionaries = []
        rawJSON = None                  # guaranteed to run once
        while rawJSON is None:
            
            try:
                sleep(random.uniform(0.5,3))

                searchResults = requests.request(
                    "GET",
                    URL,
                    headers=HEADERS,
                    params=defaultParams,
                    # proxies=proxies
                    )

                # get json version
                rawJSON = searchResults.json()

                # save to disk and to list
                with open(f'{raw_data_location}/{today}_JSON.txt', 'a') as f:
                    f.write(json.dumps(rawJSON))
                listOfTripDictionaries.extend(rawJSON['trips'])
                    
    


                # iterate pages
                while 'next_cursor' in rawJSON:
                    sleep(random.uniform(0.5,1))
    
                    page = rawJSON['next_cursor']
                    newParams = deepcopy(defaultParams)
                    newParams['from_cursor'] = page
                    
    
                    searchResults = requests.request(
                        "GET",
                        URL,
                        headers=HEADERS,
                        params=newParams, 
                        timeout=30,
                        # proxies=proxies
                    )
    
                    # get json version
                    rawJSON = searchResults.json()

                    # save to disk and to list
                    with open(f'{raw_data_location}/{today}_JSON.txt', 'a') as f:
                        f.write(json.dumps(rawJSON))
                    listOfTripDictionaries.extend(rawJSON['trips'])
    

                tripsOfOneOriginToAllDestinations.append(tuple([
                    destinationDepNum, 
                    datetime.fromtimestamp(time()), 
                    listOfTripDictionaries
                    ]))

                
    



            except ValueError:  # includes simplejson.decoder.JSONDecodeError
                print(f'Decoding JSON has failed for trips from {destinationCityRow.Commune}')

                # dump what you got
                with open(f'{raw_data_location}/{today}_JSON.txt', 'a') as f:
                    f.write('ValueError')
                    
                tripsOfOneOriginToAllDestinations.append(tuple([
                    destinationDepNum, 
                    datetime.fromtimestamp(time()), 
                    None,
                    ]))
    
            except KeyError as e:
                remaining_calls = searchResults.headers['x-ratelimit-remaining-day']
                print(e, f'with KEY {KEY}. Remaining calls: {remaining_calls}')

                if remaining_calls != 0:  
                    sleep(15)
                if remaining_calls == 0:
                    sleep(15)
                    KEY = APIdict['aux1']
                    defaultParams['key'] = KEY
                continue
                
            except ConnectionError as e:
                print(e)
                pass

    return tripsOfOneOriginToAllDestinations


#######################################################################################################################################




#%% get and prepare french prefectures

# read from human-readable excel file
frenchPrefectures  = pd.read_excel('hotels-de-prefectures-fr.xlsx', index_col=0)


# select only continental prefectures (ignore islands and colonies with alphanumeric number and larger than 100)
frenchPrefectures['DeptNum'] = pd.to_numeric(frenchPrefectures['DeptNum'], errors='coerce')
frenchPrefectures = frenchPrefectures.loc[
    (frenchPrefectures['DeptNum'].notnull()) &
    (frenchPrefectures['DeptNum'] < 100)
    ]

# cleaning up commune and deptnum cols, concatenate coords
frenchPrefectures = frenchPrefectures.assign(
    Commune=lambda df: df['Commune'].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8'),         # re-encode because of fancy french accent grave etc        
    DeptNum=lambda df: df['DeptNum'].astype(int), 

    coord=lambda df: df['LatDD'].round(4).astype(str) + ',' + df['LonDD'].round(4).astype(str),
    ).set_index('DeptNum')

frenchPrefectures


#%% request trips only starting in biggest cities (but to all prefectures)
majorCitiesList = [
    'Paris',
    'Marseille',
    'Lyon',
    'Toulouse',
    'Nice',
    'Rennes',
    'Lille',
    'Bordeaux',
    'Strasbourg',
    'Limoges',
    # 'Nimes',
    # 'Nantes',
    # 'Saint-Etienne',
]

majorCities = frenchPrefectures[frenchPrefectures.Commune.isin(majorCitiesList)].copy()
majorCities





#%% ####################################################### scrape trips ################################################################################
'''
trips_list now becomes a list of lists of lists:
[
    [origin city number (eg. PARIS),    [
                                            (destination number (eg of nice),       date,   [trip-PARIS-to-nice1, trip-PARIS-to-nice2, trip-PARIS-to-nice3, trip-PARIS-to-nice4, etc]),
                                            (destination number (eg of marseille),  date,   [trip-PARIS-to-marseille1, trip-PARIS-to-marseille2, trip-PARIS-to-marseille3, trip-PARIS-to-marseille4, etc]),
                                            ...,
                                        ]
    ], 

    [origin city number (eg. LYON),     [
                                            (destination number (eg of nice),       date,   [trip-LYON-to-nice1, trip-LYON-to-nice2, trip-LYON-to-nice3, trip-LYON-to-nice4, etc]),
                                            (destination number (eg of marseille),  date,   [trip-LYON-to-marseille1, trip-LYON-to-marseille2, trip-LYON-to-marseille3, trip-LYON-to-marseille4, etc]),
                                            ...,
                                        ]
    ], 

    [...], 
    [...], 
]
'''
trips_list = []


majorCitiesCopy = majorCities.copy()
while not majorCitiesCopy.empty:
    for originDepartmentNumber, originCityRow in majorCitiesCopy.iterrows():

        try:
            results = getTrips(
                origin=originCityRow,
                destinations=frenchPrefectures,  
                startdate=today,
                raw_data_location='01_data-raw_JSON_responses',
            )
            
            trips_list.append([originDepartmentNumber, results])
            majorCitiesCopy.drop(originDepartmentNumber, inplace=True)
        
        # Key error includes empty majorCitiesCopy. Break outside while loop
        except KeyError as e:
            print(e)
            break
        
        # If any other error, continues iterrows 
        except Exception as e:
            print(e)
            pass

        break

    cur_length = majorCitiesCopy.shape[0]
    scraped_length = len(trips_list)
    print(f"TRIP LOOP COMPLETED: Retry {cur_length} trips. {scraped_length} trips have been scraped.")

    break




pass




























#%% ####################################################### processing ################################################################################
API_results = pd.DataFrame(trips_list, columns=['DeptNum', 'results'])
API_results.set_index('DeptNum', inplace=True)

majorCities = majorCities.merge(
    API_results,
    left_index=True,
    right_index=True,
    how='left'
    )


# split trips to different departments
results = (
    majorCities
    .copy()
    .explode('results')
    )


# split destination and trip information
results[['destination', 'API_scrape_time', 'trips']] = results['results'].apply(pd.Series)


# explode json list data for individual blablacars
results = (
    results
    .drop(columns=['results'], axis=1)
    .explode('trips')
    .assign(trips=lambda df: df.trips.fillna({i: {} for i in results.index}))
    .reset_index()
    )


# flatten json individual trip data
results = results.join(pd.json_normalize(results['trips']))
results['trip_id'] = results.link.str.extract('id=(.*)')
results['waypoints'] = results.waypoints.fillna({i: [{}, {}] for i in results.index})


# split and flatten start and endpoint information
results['start'] = [x[0] for x in results['waypoints']]
results['end'] = [x[1] for x in results['waypoints']]

results = results.join(pd.json_normalize(results['start'].tolist()).add_prefix("start."))
results = results.join(pd.json_normalize(results['end'].tolist()).add_prefix("end."))


# extract actual trip identifier (numeric)
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

results['day_counter'] = numberLatestScrapingRun() + 1

# shuffle
results = results.sample(frac=1) 


results.to_csv(f'../02_process_trips/01_data-trips_with_duplicates/{date.today()}_trips_{numberLatestScrapingRun() + 1}.csv')

