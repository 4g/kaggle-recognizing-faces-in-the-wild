from keras_facenet import FaceNet
from utils import load_data
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import normalize
import numpy as np
from tqdm import tqdm


embedder = FaceNet()

train_images, train_labels, test_images, test_labels, train_pairs, submission_pairs = load_data()

### THIS IS WRONG BECAUSE EACH LABEL HAS MULTIPLE IMAGES ### 


train_indices = {l:i for i,l in enumerate(train_labels)}
test_indices = {l:i for i,l in enumerate(test_labels)}

train_embeddings = embedder.embeddings(train_images)
train_embeddings = normalize(train_embeddings)

def batch_generator():

	sample(train_pairs, 500)


	i1 = train_indices[p1]
	i2 = train_indices[p2]
	e1 = train_embeddings[i1]
	e2 = train_embeddings[i2]


