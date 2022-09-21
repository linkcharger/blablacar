# %%
import os, re, requests, datetime
from typing import Union
from pathlib import Path

import pandas as pd
import numpy as np

from dateutil import tz
from tqdm import tqdm
tqdm.pandas()
pd.options.mode.chained_assignment = None
pd.set_option("display.max_columns", 250)
pd.set_option("display.max_rows", 50)
pd.set_option("display.max_colwidth", None)
pd.options.display.float_format = "{:.2f}".format

basedir = Path(os.environ['BLABLACAR'])
datadir = basedir / "data"
scrape_datadir = datadir / 'scraper' / 'output'
thumb_datadir = datadir / "thumbnails"
parsed_trips = scrape_datadir / 'parsed_trips'

# %%
class ThumbnailParser:
    def __init__(self, files: Union[Path, list], utype: Union[Path, list], output: Path) -> None:
        """
        Class parses any number of pickle files from the parsed output and stores
        thumbnails for drivers, passengers or/and reviewers.
        
        Parameters
        ----------
        files : parsed_output files
        utype : user types
        output : output path
        """
        assert set(utype) <= set(["drivers", "passengers", "ratings"]), (
            "Unknown user type string used. Please choose between drivers, passengers, ratings."
        )
        if not isinstance(files, list):
            files = [files]
        if not isinstance(utype, list):
            utype = [utype]

        self.files = files
        self.utype = utype
        self.output = output
        self.types = {
            "drivers": {"func": self.drivers, "data": []},
            "passengers": {"func": self.passengers, "data": []},
            "ratings": {"func": self.ratings, "data": []}
        }

    def drivers(self):
        """
        Creates drivers dataset, normalizes ID label.
        """
        self.drivers_df = (
            self.data[["num_id", "driver_id", "driver_display_name", "driver_gender", "driver_thumbnail"]]
            .drop_duplicates(subset=["driver_id", "driver_thumbnail"])
            .rename(columns={"driver_id": "ID", "driver_thumbnail": "thumbnail"})
        )
        self.types["drivers"]["data"] = self.drivers_df
    
    def passengers(self):
        """
        Creates passengers dataset, normalized ID label and explodes JSON.
        """
        self.passengers_df = self.data[["num_id", "passengers"]]
        self.passengers_df = self.passengers_df.explode("passengers")
        self.passengers_df.dropna(subset=["passengers"], inplace=True)
        self.json_passengers = pd.json_normalize(self.passengers_df.passengers)
        self.passengers_df.reset_index(inplace=True, drop=True)
        self.passengers_df = pd.merge(
            self.passengers_df[["num_id"]], 
            self.json_passengers[["id", "display_name", "gender", "thumbnail"]],
            left_index=True,
            right_index=True
        )
        self.passengers_df.rename(columns={"id": "ID"}, inplace=True)
        self.types["passengers"]["data"] = self.passengers_df

    def ratings(self):
        """
        Creates ratings dataset, normalizes ID label and explodes JSON.
        """
        self.ratings_df = self.data[["num_id", "driver_id", "ratings"]]
        self.ratings_df = self.ratings_df.explode("ratings")
        self.ratings_df.dropna(subset=["ratings"], inplace=True)
        self.ratings_df.drop_duplicates(subset=["driver_id"], keep="last", inplace=True)
        self.json_ratings = pd.json_normalize(self.ratings_df.ratings)
        self.ratings_df = self.json_ratings[
            ["sender_uuid", "sender_display_name", "sender_profil_picture"]
        ]
        self.ratings_df.drop_duplicates(subset=["sender_uuid"], keep="last", inplace=True)
        self.ratings_df.rename(columns={"sender_uuid": "ID", "sender_profil_picture": "thumbnail"}, inplace=True)
        self.types["ratings"]["data"] = self.ratings_df

    def read_data(self, file: Path):
        """
        Reads parsed_output pickle, stores date and unique trip IDs.
        """
        self.data = pd.read_pickle(file)
        self.data['file_wbs'] = re.search(('\d{4}-\d{2}-\d{2}'), str(file)).group(0)
        self.data["num_id"] = self.data.trip_id.str.extract("(\d*)-").astype('int64')

    def parser(self, row):
        """
        Requests a thumbnail and stores it in `output`.
        """
        try:
            response = requests.get(row['thumbnail'])
            if response.status_code == 200:
                with open(str(self.output) + '/' + row["ID"] + '.jpeg', 'wb') as f:
                    f.write(response.content)
        except:
            pass

    def parse_trips(self):
        """
        Filters old thumbnails and parses new ones.
        """
        for file in self.files:
            thumbs = [t[:-5] for t in os.listdir(thumb_datadir) if "jpeg" in t]
            self.read_data(file=file)
            for type in self.utype:
                try:
                    self.types[type]["func"]()
                    self.outdata = self.types[type]["data"]
                    self.outdata = self.outdata.loc[~self.outdata["ID"].isin(thumbs)]
                    self.outdata.progress_apply(lambda row: self.parser(row), axis=1)
                except KeyError:
                    print(f"No new thumbnails for day {str(file.__name__)}")
                    continue






# %%
if __name__ == "__main__":
    filesload = [x for x in os.listdir(parsed_trips) if "trip_results" in str(x)]
    for file in filesload:

        instance = ThumbnailParser(
            files=parsed_trips / file,          # pickle files from before
            utype=["ratings"],
            output=thumb_datadir
            )


        instance.parse_trips()

