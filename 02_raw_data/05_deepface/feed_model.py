#%%
import csv
import os
import time
import numpy as np

import joblib
from joblib import Parallel, delayed

from model import process_image


def generate_csv_headers(max_faces):
    base_headers = ['image', 'metadata', 'faces detected']
    race_headers = []
    gender_headers = []
    
    for i in range(0, max_faces):
        race_headers += [f'face image ratio {i+1}']
        race_headers += [f'score face detection {i+1}']
        race_headers += [f'race {i+1}']
        race_headers += [f'accuracy race prediction {i+1}']
        gender_headers += [f'accuracy gender prediction {i+1}']
        gender_headers += [f'gender {i+1}']
    
    return base_headers + race_headers + gender_headers




n_jobs = joblib.cpu_count()
picturePath = '../04_download_profile_pictures/01_data-profile_pictures'





#%%############################################## init CSV ###############################################
with open(f'01_data-ethnicity_predictions/{np.datetime64("today")}.csv', 'w', newline='') as csv_file:

    images_path = os.listdir(picturePath)
    
    # process in parallel
    p = Parallel(n_jobs=n_jobs)
    
    for image in images_path:
        results = p(delayed(process_image)(             # model gets called
            image_path=image, 
            _dir=picturePath
            )
        )              


    # write CSV 
    max_faces = max([result.get('faces detected', 0) for result in results])
    fieldnames = generate_csv_headers(max_faces)

    output_csv = csv.DictWriter(csv_file, fieldnames=fieldnames)
    output_csv.writeheader()
    output_csv.writerows(results)

