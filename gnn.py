import numpy as np
from sklearn.feature_extraction import image
from PIL import Image
import networkx as nx
import os

test_path = 'data/cifar-10/test'
train_path = 'data/cifar-10/train'
train_folders = os.listdir(train_path)
del train_folders[0]
test_folders = os.listdir(test_path)
del test_folders[0]

for tf in train_folders:
    path = train_path + '/' + tf
    images = os.listdir(path)
    counter = 0
    for i in images:
        image_path = path + '/' + i
        pic = Image.open(image_path).convert('RGB')
        data = np.array(pic).astype(np.float)
        graph = image.img_to_graph(data)
        graph = nx.from_scipy_sparse_matrix(graph)
        store_path = 'data/graph-cifar-10/train' + '/' + tf + '/' + i + '.gpickle'
        nx.write_gpickle(graph, store_path)
    print("one folder done")


for tf in test_folders:
    path = test_path + '/' + tf
    images = os.listdir(path)
    counter = 0
    for i in images:
        image_path = path + '/' + i
        pic = Image.open(image_path).convert('RGB')
        data = np.array(pic).astype(np.float)
        graph = image.img_to_graph(data)
        graph = nx.from_scipy_sparse_matrix(graph)
        store_path = 'data/graph-cifar-10/test' + '/' + tf + '/' + i + '.gpickle'
        nx.write_gpickle(graph, store_path)
    print("one folder done")
