from keras_facenet import FaceNet
from utils import load_data
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import normalize
import numpy as np
from tqdm import tqdm

train_images, train_labels, test_images, test_labels, train_pairs, submission_pairs = load_data()

embedder = FaceNet()
embeddings = embedder.embeddings(test_images)
embeddings = normalize(embeddings)

print(embeddings.shape)

test_labels_indices = {l: i for i, l in enumerate(test_labels)}

distances = []

for pair in tqdm(submission_pairs):
    p1, p2 = pair
    i1 = test_labels_indices[p1]
    i2 = test_labels_indices[p2]
    e1 = embeddings[i1]
    e2 = embeddings[i2]
    dist = euclidean(e1, e2)
    distances.append(dist)

# convert distances to probability distribution
distances = np.asarray(distances)
s_dist = np.sum(distances)
p_related = []
for distance in distances:
    p_dist = 1 - np.sum(distances[np.where(distances <= distance)[0]]) / s_dist
    p_related.append(p_dist)

output_csv = open("data/output.csv", 'w')
output_csv.write("img_pair,is_related\n")

for pair, prel in zip(submission_pairs, p_related):
    s = "{pair},{prel}\n".format(pair="-".join(pair), prel=str(prel))
    output_csv.write(s)

output_csv.close()
