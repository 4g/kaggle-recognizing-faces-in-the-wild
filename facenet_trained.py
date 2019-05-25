from utils import load_data
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import normalize
import numpy as np
from tqdm import tqdm
from random import sample, choice
from itertools import product
from keras.layers.merge import dot
from keras.layers import Input, Dense
from keras.models import Model, load_model
from keras.optimizers import SGD
from keras.regularizers import l2
from itertools import  cycle
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from random import shuffle

#### Create network #####

input1 = Input(shape=(512,))
input2 = Input(shape=(512,))
dense = Dense(units=512, use_bias=True)
d1 = dense(input1)
d2 = dense(input2)
cos = dot(inputs=[d1, d2], axes=1, normalize=True)
model = Model(inputs=[input1, input2], outputs=cos)
model.compile(loss='mse', metrics=['accuracy'], optimizer=SGD(lr=0.001, momentum=0.9, nesterov=True, decay=1e-6))
model.summary()

embedder = Model(inputs=[input1], outputs=d1)
embedder.summary()
#######################3

train_images, train_labels, test_images, test_labels, train_pairs, submission_pairs = load_data()

def get_all_pairs():
    train_pairs_set = set(train_pairs)
    person_to_indices = {}

    for index, label in enumerate(train_labels):
        person_to_indices[label] = person_to_indices.get(label, [])
        person_to_indices[label].append(index)

    person_set = list(set(person_to_indices.keys()))

    def get_random_unrelated_person(e):
        x = choice(person_set)
        while ((e, x) in train_pairs_set) or ((x, e) in train_pairs_set):
            x = choice(person_set)
        return x

    repeats = lambda c, n: [c for i in range(len(n))]
    indices = []
    labels = []
    for e1, e2 in train_pairs:

        if e1 not in person_to_indices or e2 not in person_to_indices:
            continue

        e1i = person_to_indices[e1]
        e2i = person_to_indices[e2]

        r1 = get_random_unrelated_person(e1)
        r2 = get_random_unrelated_person(e2)
        r1i = person_to_indices[r1]
        r2i = person_to_indices[r2]

        e1e1 = list(product(e1i, e1i))
        e2e2 = list(product(e2i, e2i))

        e1e2 = list(product(e1i, e2i))

        e1r1 = list(product(e1i, r1i))
        e2r2 = list(product(e2i, r2i))

        indices += e1e1 + e2e2 + e1e2 + e1r1 + e2r2
        labels += repeats(1.0, e1e1) + repeats(1.0, e2e2) + repeats(0.5, e1e2) + repeats(0.0, e1r1) + repeats(0.0, e2r2)

    return indices, labels

def batch_generator(indices, labels, embeddings, batch_size):
    i1 = []
    i2 = []
    l = []
    i = 0
    for index, label in cycle(zip(indices, labels)):
        i += 1
        if i % 10 == 0:
            continue

        i1.append(embeddings[index[0]])
        i2.append(embeddings[index[1]])
        l.append(label)

        if len(l) == batch_size:
            yield [np.asarray(i1), np.asarray(i2)], np.asarray(l)
            i1, i2, l = [], [], []

    if l:
        return i1, i2, l

def val_generator(indices, labels, embeddings, batch_size):
    i1 = []
    i2 = []
    l = []
    i = 0
    for index, label in cycle(zip(indices, labels)):
        i += 1
        if i % 10 != 0:
            continue

        i1.append(embeddings[index[0]])
        i2.append(embeddings[index[1]])
        l.append(label)

        if len(l) == batch_size:
            yield [np.asarray(i1), np.asarray(i2)], np.asarray(l)
            i1, i2, l = [], [], []

    if l:
        return i1, i2, l





indices, labels = get_all_pairs()

_ind = list(range(len(indices)))
shuffle(_ind)


p1size = int(len(indices)*0.9)

indices_1, indices_2 = [indices[i] for i in sorted(_ind[:p1size])], [indices[i] for i in sorted(_ind[p1size:])]
labels_1, labels_2 = [labels[i] for i in sorted(_ind[:p1size])], [labels[i] for i in sorted(_ind[p1size:])]

batch_size = 128

print("Num siamese pairs", len(indices))

from keras_facenet import FaceNet
face_embedder = FaceNet()
train_embeddings = face_embedder.embeddings(train_images)
train_embeddings = normalize(train_embeddings)

generator = batch_generator(indices=indices_1, labels=labels_1, embeddings=train_embeddings,
                                    batch_size=batch_size)

val = val_generator(indices=indices_2, labels=labels_2, embeddings=train_embeddings,
                                    batch_size=batch_size)

def lr_schedule(epoch):
    if epoch <= 30:
        lr = .001
    else:
        lr = 0.0001

    return lr

lr_scheduler = LearningRateScheduler(schedule=lr_schedule, verbose=1)
checkpoint = ModelCheckpoint(filepath="best_model.hf5", monitor='val_loss', save_best_only=True, mode='auto')

model.fit_generator(generator,
                    steps_per_epoch=len(indices_1)//batch_size,
                    epochs=60,
                    validation_data=val,
                    validation_steps=len(indices_2)//batch_size, callbacks=[lr_scheduler, checkpoint])


model = load_model("best_model.hf5")

test_labels_indices = {l: i for i, l in enumerate(test_labels)}
distances = []

face_embeddings = face_embedder.embeddings(test_images)
embeddings = normalize(face_embeddings)
embeddings = embedder.predict(embeddings)

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
