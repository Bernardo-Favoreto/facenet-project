#import face_recognition
import os
import pickle
import numpy as np
from keras.models import load_model
from mtcnn.mtcnn import MTCNN
from embeddings import*
import argparse

parser = argparse.ArgumentParser(description='True se você quiser atualizar os arquivos pickle')
parser.add_argument('--update', type=bool, default=False)
options = parser.parse_args()


if options.update == True:
    update_or_create_known_people()


model = load_model('facenet_keras.h5', compile=False)
known_face_encodings = []
known_face_names = []
def update_or_create_known_people():
    """
    This will update the names and encodings pickle file.
    """
    for name in os.listdir('fotos'):
        identity = ''.join(filter(str.isalpha, name[:-4]))
        encoded_img = get_embedding(model, extract_face('fotos/'+str(name)))
        known_face_encodings.append(encoded_img)
        known_face_names.append(identity)
    save_pickle(known_face_names, "names.pickle")
    save_pickle(known_face_encodings, "encodings.pickle")
    print("Pickle files sucessfuly generated/updated!")


def save_pickle(item, pickle_name):
    pickle_out = open(pickle_name, "wb") #Write bytes
    pickle.dump(item, pickle_out)
    pickle_out.close()

def load_pickle(pickle_file):
    pickle_in = open(pickle_file, "rb") #Read bytes
    pickle_file = pickle.load(pickle_in)
    return pickle_file

def avg_array(list_of_arrays):
    b = np.zeros(list_of_arrays[0].shape)
    for array in list_of_arrays:
        b += array
    return b/len(list_of_arrays)


def distance(emb_base, emb_list, names_list):
    dist = {}
    print(len(names_list))
    for i in range(len(emb_list)):
        identity = names_list[i]
        embedding = emb_list[i]
        dist[identity] = np.linalg.norm(emb_base - embedding)

    return dist

