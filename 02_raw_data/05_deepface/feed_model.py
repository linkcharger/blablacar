#%%
import os, shutil, time, zipfile, csv, time, sys, argparse, json, joblib
from joblib import Parallel, delayed
from model import process_image

parser = argparse.ArgumentParser()
parser.add_argument("zip")
args = parser.parse_args()



def generate_csv_headers(max_faces):
    base_headers = ['image', 'metadata', 'faces detected']
    race_headers = []
    gender_headers = []
    for i in range(0, max_faces):
        race_headers += ['face image ratio %d' % (i+1)]
        race_headers += ['score face detection %d' % (i+1)]
        race_headers += ['race %d' % (i+1)]
        race_headers += ['accuracy race prediction %d' % (i+1)]
        gender_headers += ['accuracy gender prediction %d' % (i+1)]
        gender_headers += ['gender %d' % (i+1)]
    return base_headers + race_headers + gender_headers




n_jobs = joblib.cpu_count()
timestamp = int(time.time()*1000000)
images_dir_path = 'tmp/%s/' % timestamp
images_zip_path = 'data/%s' % args.zip







#%%######################################################################################################
print('extracting images...')

with zipfile.ZipFile(images_zip_path, 'r') as images_zip:
    images_zip.extractall(images_dir_path)







#%%############################################## init CSV ###############################################
print('processing...')
csv_file_name = '%d.csv' % int(time.time())


with open('data/%s' % csv_file_name, 'w', newline='') as csv_file:

    images_path = os.listdir(images_dir_path)
    # Process images

    p = Parallel(n_jobs=n_jobs)
    for image in images_path:
        results = p(delayed(process_image)(image_path=image, _dir=images_dir_path))              # model gets called


    # Write CSV headers
    max_faces = max([result.get('faces detected', 0) for result in results])
    fieldnames = generate_csv_headers(max_faces)
    output_csv = csv.DictWriter(csv_file, fieldnames=fieldnames)
    output_csv.writeheader()
    output_csv.writerows(results)





# Cleanup
shutil.rmtree(images_dir_path)

print('Generated %s' % csv_file_name)
