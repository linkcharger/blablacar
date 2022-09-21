# %%
import os
import re
import requests
import datetime
from pathlib import Path

import pandas as pd
import numpy as np
import plotly
import plotly.express as px
import plotly.figure_factory as ff
# from IPython.core.display import display, Markdown
import texpro as tp

import matplotlib.pyplot as plt
import seaborn as sns

from dateutil import tz
from tqdm import tqdm
tqdm.pandas()

pd.set_option("display.max_columns", 250)
pd.set_option("display.max_rows", 50)
pd.set_option("display.max_colwidth", None)
pd.options.display.float_format = "{:.2f}".format
plotly.offline.init_notebook_mode(connected=True)

basedir = Path(os.environ['BLABLACAR'])
datadir = basedir / "output_data" / "trips_ethnicities"
scrape_datadir = basedir / "output_data" / 'scraper' / 'output'
thumb_datadir = basedir / "output_data" / "thumbnails"
parsed_trips = scrape_datadir / 'parsed_trips'
facedir = basedir / "deepface"

os.chdir(basedir)
# %%

COLUMNS = {
    "passenger": [
        "pass_total", 
        "pass_total_obs", 
        "pass_asian", 
        "pass_asian_pc", 
        "pass_black", 
        "pass_black_pc", 
        "pass_indian", 
        "pass_indian_pc", 
        "pass_latino", 
        "pass_latino_pc", 
        "pass_mideast", 
        "pass_mideast_pc", 
        "pass_norace", 
        "pass_norace_pc", 
        "pass_white", 
        "pass_white_pc"
        ],

    "driver" : [
        "driv_total", 
        "driv_total_obs", 
        "driv_asian", 
        "driv_asian_pc", 
        "driv_black", 
        "driv_black_pc", 
        "driv_indian", 
        "driv_indian_pc", 
        "driv_latino", 
        "driv_latino_pc", 
        "driv_mideast", 
        "driv_mideast_pc", 
        "driv_norace", 
        "driv_norace_pc", 
        "driv_white", 
        "driv_white_pc"
        ]
}

def ethnicity_parse(row, role: str):
    """
    For each driver, compute the average share of passengers they accepted with xxx ethnicity.
    What changes is the denominator:
        - for '_total' variables, the denominator is the number of people in the driver's ratings (eg. 10% of the ppl that reviewed them (from rides past) are black)
        - for '_total_obs' variables, the denominator is the number of people we observe having been passengers of this driver (eg. 5% or 20% of the people we directly observe are black)

    We have to remember that when we scrape, we are essentially taking a subset of all rides ever taken. 
    Therefore, the ethnicities based on this subset might in some way deviate (stochastically or structurally) from the complete set of rides. 
    Thus, if nothing changed structurally (before and during scraping), we would expect the ratings based on the complete set to be less biased and more precise.
    """
    ind_ratings = pd.json_normalize(row["ratings"])
    try:
        ind_ratings = ind_ratings.loc[lambda df: df.sender_uuid!=""]

        ind_ratings.dropna(subset=["sender_uuid"], inplace=True)
    except:
        pass


    if ind_ratings.shape[0] > 0 and "role" in ind_ratings:
        lst_ = ind_ratings.loc[ind_ratings["role"]==role]["sender_uuid"].tolist()
        f = (
            ratings_eth
            .loc[ratings_eth["ID"].isin(lst_)]
            .dropna(subset=["ethnicity"])
        )
        total = len(lst_)
        total_obs = f.shape[0]

        if f.shape[0] == 0:
            total = 1
        asian = f.ethnicity.str.count("Asian").sum()
        black = f.ethnicity.str.count("Black").sum()
        indian = f.ethnicity.str.count("Indian").sum()
        latino = f.ethnicity.str.count("Latino_Hispanic").sum()
        mideast = f.ethnicity.str.count("Middle Eastern").sum()
        norace = f.ethnicity.str.count("No Race").sum()
        white = f.ethnicity.str.count("White").sum()

        asian_pc = f.ethnicity.str.count("Asian").sum() / total
        black_pc = f.ethnicity.str.count("Black").sum() / total
        indian_pc = f.ethnicity.str.count("Indian").sum() / total
        latino_pc = f.ethnicity.str.count("Latino_Hispanic").sum() / total  
        mideast_pc = f.ethnicity.str.count("Middle Eastern").sum() / total
        norace_pc = (f.ethnicity.str.count("No Race").sum() + total - total_obs) / total
        white_pc = f.ethnicity.str.count("White").sum() / total

    elif ind_ratings.shape[0] == 0 or "role" not in ind_ratings:
        total, total_obs, asian, black, indian, latino, mideast, norace, white = 0, 0, 0, 0, 0, 0, 0, 0, 0
        asian_pc, black_pc, indian_pc, latino_pc, mideast_pc, norace_pc, white_pc = (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

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


# %%
predictions = pd.read_csv(datadir / "ethnicity_predictions.csv")
filesload = [x for x in os.listdir(parsed_trips) if "trip_results" in str(x)]
filesparsed = [x for x in os.listdir(datadir / "day_trips") if "ethnicity_trips" in str(x)]




# %%
for item in tqdm(filesload):
    if not [x for x in filesparsed if re.search(('\d{4}-\d{2}-\d{2}'), item).group(0) in x]:
        day_file = []
        for role in ["passenger", "driver"]:
            scrape_df = pd.read_pickle(parsed_trips / item)
            scrape_df['file_wbs'] = re.search(('\d{4}-\d{2}-\d{2}'), item).group(0)
            scrape_df["num_id"] = scrape_df.trip_id.str.extract("(\d*)-").astype('int64')
            ratings = scrape_df[["num_id", "driver_id", "ratings"]]
            ratings_df = ratings.drop_duplicates(subset=["driver_id"], keep="last", inplace=False)
            ratings_df = ratings_df.explode("ratings")
            ratings_df.dropna(subset=["ratings"], inplace=True)

            json_ratings = pd.json_normalize(ratings_df.ratings)

            ratings_df = json_ratings[["sender_uuid", "sender_display_name", "sender_profil_picture"]]
            ratings_df.drop_duplicates(subset=["sender_uuid"], keep="last", inplace=True)
            ratings_df.rename(columns={"sender_uuid": "ID", "sender_profil_picture": "thumbnail"}, inplace=True)

            ratings_eth = pd.merge(ratings_df, predictions, on="ID", how="left", indicator=False)

            trips = ratings.drop_duplicates("num_id")
            trips[COLUMNS[role]] = trips.progress_apply(lambda row: ethnicity_parse(row, role=role), axis=1, result_type="expand")
            trips = trips.drop("ratings", axis=1)
            day_file.append(trips)

        trips = pd.merge(day_file[0], day_file[1], on=["num_id", "driver_id"])
        trips.to_csv(datadir / "day_trips" / str("ethnicity_trips_" + re.search(('\d{4}-\d{2}-\d{2}'), item).group(0) + ".csv"))
        