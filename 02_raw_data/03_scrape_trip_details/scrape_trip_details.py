#%%
import json
import logging
import random
import time
from concurrent.futures import (ALL_COMPLETED, ThreadPoolExecutor, as_completed, wait)
from datetime import date, datetime, timedelta
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# from requests.packages.urllib3.exceptions import InsecureRequestWarning
# requests.packages.urllib3.disable_warnings(InsecureRequestWarning)


########################################################################################################################

class ScrapeSession(object):
    _BASE_URL = 'https://www.blablacar.co.uk'
    USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.93 Safari/537.36 OPR/76.0.4017.107'
    
    def __init__(self):
        self._logger = logging.getLogger(self.__class__.__name__)
        
        # ua = UserAgent()
        # USER_AGENT = str(self.ua.random)
        
        time.sleep(random.uniform(1, 2))
        
        self._create_session()
        
    @staticmethod
    def _super_proxy():
        url = 'https://wtfismyip.com'
        proxy_host = 'gate.smartproxy.com'
        proxy_port = '7000'
        proxy_user = 'blablacar'
        proxy_password = 'blablacar_pass'

        proxies = {
            'https': f'http://user-{proxy_user}:{proxy_password}@{proxy_host}:{proxy_port}/',
            'http': f'http://user-{proxy_user}:{proxy_password}@{proxy_host}:{proxy_port}/',
        }
        
        return proxies

    def _create_session(self):
        headers = {
            'User-Agent': self.USER_AGENT,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en_GB',
            'Accept-Encoding': 'gzip, deflate',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Pragma': 'no-cache',
            'Cache-Control': 'no-cache',
        }
        
        self.session = requests.session()
        self.session.headers = headers
        self.session.verify = False
        proxy = self._super_proxy()
        self.session.proxies = proxy
        
        self.skip = False
        while True:
            try:
                self.session.get(self._BASE_URL, timeout=10)
                time.sleep(random.uniform(1,3))
                break
            except Exception as e:
                print('ERROR AT SETTING SESSION:', e)
                self.skip = True
                break
        
    def scrape(self, trip_id):
        '''
        Parses JSON results from trip-specific Blablacar page. Returns 
            - Name of driver
            - Description
            - Passenger' names
            - Driver reviews
        
        Returns
        -------
        :param trip_id:
        :return:
            
        '''
        trip_info = {'trip': trip_id}
        self._logger = logging.LoggerAdapter(self._logger, trip_info)
        
        result = {
            trip_id: {
                'ride': {},
                'rating': [],
                'status': None,
                }
            }
        
        data = {
            'source': 'CARPOOLING',
            'id': trip_id,
            }
        
        i = 0
        
        self._logger.info(f'CREATE SESSION ({self.session.proxies["http"]})')
        




        while True:    
            try:
                # Check for Session set-up
                if self.skip:
                    result[trip_id]['status'] = False
                    break
                
                # Loop iteration
                i+=1
                
                # If the scrape fails two times, skip
                if i >= 2:
                    self._logger.info('SKIPPED REQUEST')
                    result[trip_id]['status'] = False
                    time.sleep(random.uniform(10,60))
                    break
                
                time.sleep(random.uniform(2,6))

                ############################################################# just call the normal website to get cookies ##################################################################################

                self._logger.info(f'REQUESTING BASIC INFO ({self.session.proxies["http"]})')


                response = self.session.get(
                    f'{self._BASE_URL}/trip',
                    params=data,
                    timeout=30
                )
                time.sleep(random.uniform(1,5))



                # Catch FORBIDDEN HTML responses
                if response.status_code == 403:
                    self._logger.info(f'403 FORBIDDEN ERROR ({self.session.proxies["http"]}): {response.reason}')
                    time.sleep(random.uniform(4,6))
                    
                    # If repeated, break code; else return to while loop
                    if i >= 2:
                        self._logger.info('SKIPPED REQUEST')
                        time.sleep(random.uniform(4,6))
                        result[trip_id]['status'] = False
                        break
                    continue
                
                # Catch Bad Gateway responses
                if response.status_code == 502: # Not an exception
                    self._logger.info(f'502 BAD GATEWAY: {response.reason}')
                    time.sleep(random.uniform(5, 40))
                    
                    # If repeated, break code; else return to while loop
                    if i >= 2:
                        self._logger.info('GATEWAY LOSS')
                        result[trip_id]['status'] = 'Deleted'
                        break
                    continue
                    
                time.sleep(random.uniform(4,8))
                

                ############################################################## request ride info through edge endpoint #######################################################################################################
                

                self._logger.info('REQUESTING TRIP DETAILS')
                
                self.session.headers = {
                    'User-Agent': self.USER_AGENT,
                    'Accept': 'application/json',
                    'Accept-Language': 'en_GB',
                    'Accept-Encoding': 'gzip, deflate, br',
                    'Referer': str(self._BASE_URL +'/'),
                    'Content-Type': 'application/json',
                    'sec-ch-ua-mobile': '?0', #
                    'sec-fetch-dest': 'empty', #
                    'sec-fetch-mode': 'cors', #
                    'sec-fetch-site': 'same-site', #
                    'X-Blablacar-Accept-Endpoint-Version': '4', # 5
                    'x-locale': 'en_GB',
                    'x-visitor-id': self.session.cookies['vstr_id'],
                    'x-currency': 'GBP',
                    'x-client': 'SPA|1.0.0',
                    'x-forwarded-proto': 'https',
                    'Authorization': f'Bearer {self.session.cookies["app_token"]}',             # use previously acquired cookies
                    'Origin': 'https://www.blablacar.co.uk',
                    'DNT': '1',
                    'Connection': 'keep-alive',
                    'Pragma': 'no-cache',
                    'Cache-Control': 'no-cache',
                    'TE': 'Trailers'
                    }
                
                data = {
                    'source': 'CARPOOLING',
                    'id': trip_id,
                    'requested_seats': '1',
                    }
                
                response = self.session.get(
                    'https://edge.blablacar.co.uk/ride',
                    params=data,
                    timeout=30
                    )
                

                # Capture deleted trips between API call and web scrape
                if response.status_code == 404:
                    self._logger.info(f'TRIP DELETED: {response.reason}')
                    time.sleep(random.uniform(2,3))
                    result[trip_id]['status'] = 'Deleted'
                    break
                
                # Capture any other exceptions; return control to while loop
                if not response.ok:
                    self._logger.info(f'FAULT AT SECOND REQUEST: {response.status_code} {response.reason}')
                    continue
        
                if response.status_code == 502: # Not an exception
                    self._logger.info(f'502 BAD GATEWAY: {response.reason}')
                    time.sleep(random.uniform(5, 10))
                            
                    # If repeated, break code; else return to while loop
                    if i >= 2:
                        self._logger.info('GATEWAY LOSS')
                        result[trip_id]['status'] = 'Deleted'
                        break
                    continue
                
                ride = response.json()
                result[trip_id]['ride'] = ride
                
                time.sleep(random.uniform(4,6))
                
                ######################################################## request reviewer info through EDGE and old API ###########################################################
                
                self._logger.info('REQUESTING RATINGS')
                
                page_num = 0
                total_pages = 1
                
                # Loop over all ratings pages
                while page_num < total_pages:
                    page_num += 1
                    data = {
                        'page': page_num,
                        'limit': '100',
                    }
                    
                    response = self.session.get(
                        f'https://edge.blablacar.co.uk/api/v2/users/{ride["driver"]["id"]}/rating',
                        params=data,
                        timeout=30
                    )
                    
                # Capture any exceptions; return control to while loop
                    if not response.ok:
                        self._logger.info(f'FAULT AT THIRD REQUEST: {response.status_code} {response.reason}')
                        continue
                
                    ratings_data = response.json()          # includes reviewer data
                    
                    try:
                        result[trip_id]['rating'].append(ratings_data['ratings'])
        
                        total_pages = ratings_data['pager'].get('pages', total_pages)
                    
                        time.sleep(random.uniform(4,6))
                    
                        self._logger.info('RATINGS PAGE %s' % page_num)
                    
                    except KeyError:
                        self._logger.info('NO RATINGS')
                        result[trip_id]['rating'] = ['No Ratings']
                        result[trip_id]['web_scrape_time'] = round(time.time())
                


                # Successful scrape; break with status True
                self._logger.info('<<<FINISHED SCRAPE>>>')
                result[trip_id]['status'] = True
                result[trip_id]['web_scrape_time'] = round(time.time())
                break
            


            # Capture any other exceptions; return control to while loop
            except Exception as e:
                self._logger.info(f'REQUEST ERROR: {e}')
                time.sleep(random.uniform(6,8))
                continue
        
        return result
    
########################################################################################################################

now = datetime.now()

today = np.datetime64('today')
tomorrow = today + np.timedelta64(1, 'D')

outputFile = f'01_data-raw_JSON_trip_details_{today}_trips.txt'   


###### logging ######
MESSAGE_INFO = '%(asctime)s %(trip)s ----- %(message)s'
DATEFMT = '%Y/%m/%d %H:%M'
file_handler = logging.FileHandler(
    filename=f'logs/{today}_scraper.log', 
    mode='a',
    )

file_handler.setFormatter(
    logging.Formatter(
        fmt=MESSAGE_INFO,
        datefmt=DATEFMT,
        )
    )

stream_handler = logging.StreamHandler()

logging.basicConfig(
    level=logging.INFO,
    format=MESSAGE_INFO,
    datefmt=DATEFMT,
    handlers=[
        file_handler,
        stream_handler,
        ]
    )












#%% #################################################### PROCESSING ####################################################################################################
todaysSearchResultsFiles = glob(f'../02_process_trips/01_data-trips_with_duplicates/{today}*.csv')                               # '_API_dumps' / 'csv' / [date]_trips_0.csv

# create dataframe of all observations of today
lst_results = []
for item in todaysSearchResultsFiles:
    _ = pd.read_csv(item)
    lst_results.append(_)
results = pd.concat(lst_results)        

# Define datetimes
results['start.date_time'] = pd.to_datetime(results['start.date_time'])
results['end.date_time'] = pd.to_datetime(results['end.date_time'])

# Save unique matches for all trips
store_results = (
    results
    .dropna(subset=['trip_id'])
    .sort_values(by=['num_id', 'day_counter'])                                          # default: ascending 
    .drop_duplicates(subset=['num_id', 'DeptNum', 'destination'], keep='first')         # take the FIRST
)


store_results.to_csv(f'../02_process_trips/02_data-trips_without_duplicates/{datetime.now().strftime("%Y%m%d_%H")}h_trips.csv')    












################################################## GET TRIP IDS ##################################################
# Preserve last time a trip is scraped in any of the 5 daily loops, while retaining information on the iteration at which it's scraped for the first time
todaysSearchResults = (
    results
    .dropna(subset=['trip_id'])
    .sort_values(by=['num_id', 'day_counter'])
    .assign(iter_found=lambda df: df.groupby('num_id')['day_counter'].transform('min'))
    .drop_duplicates(subset=['num_id'], keep='last')
    )


# Keep today-tomorrow trips
todaysSearchResults = (
    todaysSearchResults
    .loc[
        (todaysSearchResults['start.date_time'].dt.date == today) 
        # | (API_results['start.date_time'].dt.date == tomorrow)
    ]
)



# Create list to run web scraper through -> this is so, so bad - using the same variable for a different type AND overwriting it
tripIDlist = todaysSearchResults['trip_id'].to_list()
tripIDlist[:10]








#%% ################################################## scrape trip details ##################################################
trips_dict = {}
json_dump = []

while tripIDlist:

    try:
        base_len = len(tripIDlist)

        threads = []
        with ThreadPoolExecutor(max_workers=6) as executor:

            for tripID in tripIDlist:
                threads.append(executor.submit(ScrapeSession().scrape, tripID))
                time.sleep(random.uniform(3, 5))

            for tripID in as_completed(threads):             # 03_scrape_trip_details/01_data-raw_json_dumps/
                json_dump.append(tripID.result())

            wait(threads, timeout=7200, return_when=ALL_COMPLETED)

        merged_results = [
            x 
            for thread in threads 
            for x in thread.result().items()                        # this is the dictionary that 'ScrapeSession().scrape() returns
            ]
        trips_dict.update(dict((x, y) for x, y in merged_results))  # here we separate it again into key-value pairs




        tripIDlist = [x for x in trips_dict if not trips_dict[x]['status']]     # take only the tripIds which have status 'false'

        next_len = len(tripIDlist)
        print(f'ITERATION COMPLETED. NEXT ITERATION HAS {next_len}, DOWN FROM {base_len} ORIGINAL TRIPS')


    except Exception as e:
        print('########### ERROR THAT TERMINATES WHILE LOOP', e)
        # If crash, save results
        with open(outputFile, 'w') as f:       # 03_scrape_trip_details/01_data-raw_json_dumps/
            f.write(json.dumps(json_dump))
            


# Dump results if trip id's are exhausted
with open(outputFile, 'w') as f:            # 03_scrape_trip_details/01_data-raw_json_dumps/
    f.write(json.dumps(json_dump))        





# #%% Parse JSON data
# output_df = parser(file_to_operate)         # now in 03_scrape_trip_details/01_data-raw_json_dumps/
# file_to_save = uniquifier(str(outdir / 'parsed_trips' / f'{today}_parsed_trip_results.pkl'))
# output_df.to_pickle(file_to_save)       # now in 03_scrape_trip_details/02_data-final_output_pickles/

