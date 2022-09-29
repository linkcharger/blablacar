import os, cv2, csv, time, json
import numpy as np

from keras_facenet import FaceNet

import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Activation
from tensorflow.keras.layers import Conv2D, AveragePooling2D
from tensorflow.keras.models import Model, Sequential

import PIL.Image
import PIL.ExifTags



races_order = ['Asian', 'Indian', 'Black', 'Middle Eastern', 'White', 'Latino_Hispanic']
genders_order = ['Male', 'Female']

embedder = FaceNet()

# Calibration dics
calibration_dic_races = np.load('deepface_copia/model/calibration_dic_races.npy', allow_pickle=True).item()
calibration_dic_genders = np.load('deepface_copia/model/calibration_dic_genders.npy', allow_pickle=True).item()

def calibrate(calibration_lst, theta):
    return calibration_lst[int(theta*len(calibration_lst))]

def cropbox(image, detection, margin):
    x1, y1, w, h = detection
    x1 -= margin
    y1 -= margin
    w += 2*margin
    h += 2*margin
    if x1 < 0:
        w += x1
        x1 = 0
    if y1 < 0:
        h += y1
        y1 = 0
    return image[y1:y1+h, x1:x1+w]

def get_image_metadata(image_path):
    try:
        img = PIL.Image.open(image_path)
        exif = { PIL.ExifTags.TAGS[k]: str(v) for k, v in img._getexif().items() if k in PIL.ExifTags.TAGS }
        return exif
    except:
        return {}
        
# Model
print('Loading model...')
inputs = Input((512,))
model = Dense(300, activation='relu')(inputs)
model = tf.keras.layers.BatchNormalization()(model)
model = Dropout(0.5)(model)
model = Dense(300, activation='relu')(model)
model = tf.keras.layers.BatchNormalization()(model)
model = Dropout(0.5)(model)
model = Dense(120, activation='relu')(model)
model = tf.keras.layers.BatchNormalization()(model)
model = Dropout(0.5)(model)
r = Dense(6, activation=None,name='race2')(model)
r = Activation('softmax',name='race')(r)
g = Dense(2, activation=None,name='gender2')(model)
g = Activation('softmax',name='gender')(g)
joint_model = Model(inputs=inputs, outputs=[r, g])
joint_model.summary()
joint_model.load_weights('deepface_copia/model/training_model.ckpt')

def process_image(image_path, _dir):

    result = {}
    result['image'] = image_path
    result['metadata'] = json.dumps(get_image_metadata('%s/%s' % (_dir, image_path)))
    # Detect faces
    image = cv2.imread('%s/%s' % (_dir, image_path))
    try:
        image_height, image_width, image_channels = image.shape
    except:
        return result        
    image_area = image_height * image_width
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    embedder_result = embedder.mtcnn().detect_faces(image)
    # Iterate faces detected and generate face_images array
    detection_confidences = []
    face_images = []
    face_image_ratios = []
    for i, face_detected in enumerate(embedder_result):
        box = face_detected.get('box', [])
        confidence = face_detected.get('confidence', 0)
        if confidence < 0.95:
            continue
        detection_confidences += [confidence]
        face_images += [cropbox(image, box, 0)]
        face_image_ratios += [box[2]*box[3]/image_area]
    # Save amount of faces detected
    result['faces detected'] = len(face_images)
    # Skip loop if no faces were detected
    if not face_images:
        return result
    try:
        # Get embeddings and predictions for faces detected
        embeddings = embedder.embeddings(np.array(face_images))
        predicted_races, predicted_genders = joint_model.predict(np.array(embeddings))
        # Generate race results
        for i, predicted_race in enumerate(predicted_races):
            result['face image ratio %d' % (i+1)] = face_image_ratios[i]
            face_detection_confidence = detection_confidences[i] * 5
            result['score face detection %d' % (i+1)] = face_detection_confidence
            selected_race_index = np.argmax(predicted_race)
            selected_race = races_order[selected_race_index]
            result['race %d' % (i+1)] = selected_race
            race_prediction_confidence = calibrate(calibration_dic_races[selected_race_index], np.max(predicted_race))
            result['accuracy race prediction %d' % (i+1)] = race_prediction_confidence
        # Generate gender results
        for i, predicted_gender in enumerate(predicted_genders):
            selected_gender_index = np.argmax(predicted_gender)
            selected_gender = genders_order[selected_gender_index]
            result['gender %d' % (i+1)] = selected_gender
            gender_prediction_confidence = calibrate(calibration_dic_genders[selected_gender_index], np.max(predicted_gender))
            result['accuracy gender prediction %d' % (i+1)] = gender_prediction_confidence
        return result
    except:
        return result
