"""
Setup:
Dropout - p=0.5
SGD lr = 0.01
fasttext for pretrained embeddings (emb_size = 300)
filter windows (h) of 3, 4, 5 with 100 feature maps each
with minibatching (batch_size = 100)
"""

from collections import defaultdict
import time
import random
import torch
import math
import copy
import numpy as np

class CNNclass(torch.nn.Module):
    def __init__(self, nwords, emb_size, num_filters, ntags, emb_weight):
        super(CNNclass, self).__init__()

        """ layers """
        emb = torch.nn.Embedding(nwords, emb_size)
        # update embeddings with pre-trained vectors
        emb.load_state_dict({'weight': emb_weight})
        self.embedding = emb
        
        # Conv 1d with filter windows of size 3, 4, 5
        self.conv_1d_3 = torch.nn.Conv1d(in_channels=emb_size, out_channels=num_filters, kernel_size=3,
                                       stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv_1d_4 = torch.nn.Conv1d(in_channels=emb_size, out_channels=num_filters, kernel_size=4,
                                       stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv_1d_5 = torch.nn.Conv1d(in_channels=emb_size, out_channels=num_filters, kernel_size=5,
                                       stride=1, padding=0, dilation=1, groups=1, bias=True)
        # Dropout (p=0.5)
        self.dropout = torch.nn.Dropout()
        self.relu = torch.nn.ReLU()
        self.projection_layer = torch.nn.Linear(in_features=num_filters*3, out_features=ntags, bias=True)
        # Initializing the projection layer
        torch.nn.init.xavier_uniform_(self.projection_layer.weight)

    def forward(self, words):
        emb = self.embedding(words)                   
        if emb.dim() == 2:
            emb = emb.unsqueeze(0)
        emb = emb.permute(0, 2, 1)
        h3 = self.conv_1d_3(emb)                      # (1 or BATCH_SIZE) x num_filters x len(words)
        h4 = self.conv_1d_4(emb)                      
        h5 = self.conv_1d_5(emb)                      
        # Do max pooling
        h3 = h3.max(dim=2)[0]                         # (1 or BATCH_SIZE) x num_filters
        h4 = h4.max(dim=2)[0]                         
        h5 = h5.max(dim=2)[0]                         
        # concatnate filters from 3 window size
        h = torch.cat((h3, h4, h5), 1) 
        # apply dropout on penultimate layer
        h = self.dropout(h)
        h = self.relu(h)
        out = self.projection_layer(h)              # size(out) = (1 or BATCH_SIZE) x ntags
        return out
        
# Functions to read in the corpus
w2i = defaultdict(lambda: len(w2i))
t2i = defaultdict(lambda: len(t2i))
UNK = w2i["<unk>"]

def read_dataset(filename):
    with open(filename, "r") as f:
        for line in f:
            tag, words = line.lower().strip().split(" ||| ")
            yield ([w2i[x] for x in words.split(" ")], t2i[tag])
            
# Part of the code in this function is taken from https://fasttext.cc/docs/en/english-vectors.html
# construct embedding weight from the pretrained vectors
def read_pretrained_emb(filename, nwords, emb_size):
    # uniform initialization for vocab not in pre-trained embeddings
    emb_weight = (np.random.rand(nwords, emb_size) - 0.5) / 2
    dict_w2i = dict(w2i)
    
    with open(filename, "r") as f:
        n, dim = map(int, f.readline().split())
        for line in f:
            tokens = line.rstrip().split(' ')
            try:
                if dict_w2i[tokens[0]]:
                    vec = list(map(float, tokens[1:]))
                    emb_weight[dict_w2i[tokens[0]]] = np.array(vec)
            except KeyError:
                pass
    return torch.from_numpy(emb_weight)

# Read in the training data
train = list(read_dataset("../topicclass/topicclass_train.txt"))
nwords = len(w2i)

w2i = defaultdict(lambda: UNK, w2i)
dev = list(read_dataset("../topicclass/topicclass_valid.txt"))
test = list(read_dataset("../topicclass/topicclass_test.txt"))
ntags = len(t2i)
i2t = {}
for tag, index in t2i.items():
    i2t[index] = tag
    
# Define the model
EMB_SIZE = 300
FILTER_SIZE = 100
WIN_SIZE = 5
BATCH_SIZE = 100

# Create embeddings with pre-trained vector
emb_weight = read_pretrained_emb("../crawl-300d-2M.vec", nwords, EMB_SIZE)

# initialize the model
model = CNNclass(nwords, EMB_SIZE, FILTER_SIZE, ntags, emb_weight)
criterion = torch.nn.CrossEntropyLoss() # softmax + NLL loss
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

type = torch.LongTensor
use_cuda = torch.cuda.is_available()

if use_cuda:
    type = torch.cuda.LongTensor
    model.cuda()

max_dev_acc = 0.0
max_train_acc = 0.0
max_iter = 0
iter_count = 0

for ITER in range(100):
    # Perform training
    random.shuffle(train)
    train_loss = 0.0
    train_correct = 0.0
    start = time.time()
    
    # minibatching
    for mb in range(math.ceil(len(train)/BATCH_SIZE)):
        batch = train[mb*BATCH_SIZE:(mb+1)*BATCH_SIZE]
        words_batch = []
        tags_batch = []
        max_len = WIN_SIZE
        for i in range(len(batch)):
            if len(batch[i][0]) > max_len:
                max_len = len(batch[i][0])
        for data in batch:
            tags_batch.append(data[1])
            words = copy.deepcopy(data[0])
            # pad 0 at the end to match longest sentence in batch
            if len(words) < max_len:
                words += [0] * (max_len - len(words))
            words_tensor = torch.tensor(words).type(type)
            words_batch.append(words_tensor)
        tags_batch_tensor = torch.tensor(tags_batch).type(type)
        words_batch_tensor = torch.stack(words_batch)
        
        scores = model(words_batch_tensor)
        _, predicted = torch.max(scores.data, 1)
        train_correct += (predicted == tags_batch_tensor).sum().item()

        my_loss = criterion(scores, tags_batch_tensor)
        train_loss += my_loss.item()
        
        # Do back-prop
        optimizer.zero_grad()
        my_loss.backward()
        optimizer.step()
    train_acc = train_correct / len(train)
    print("iter %r: acc=%.4f, time=%.2fs" % (ITER + 1, train_acc, time.time() - start))
    
    # Perform validation on dev set
    dev_correct = 0.0
    for words, tag in dev:
        # Padding (can be done in the conv layer as well)
        if len(words) < WIN_SIZE:
            words += [0] * (WIN_SIZE - len(words))
        words_tensor = torch.tensor(words).type(type)
        scores = model(words_tensor)[0]
        predict = scores.argmax().item()
        if predict == tag:
            dev_correct += 1
    dev_acc = dev_correct / len(dev)
    print("iter %r: dev acc=%.4f" % (ITER + 1, dev_acc))
    
    iter_count += 1
    # early stopping
    if train_acc >= (dev_acc - 0.003) and dev_acc > max_dev_acc:
        max_iter = ITER + 1
        max_dev_acc = dev_acc
        max_train_acc = train_acc
        iter_count = 0
        torch.save(model.state_dict(), "best_model.tar")
        print(max_iter, max_dev_acc, max_train_acc)
    if iter_count > 7 and max_dev_acc > 0.0:
        break
        
# use the best model to predict dev and test labels
model.load_state_dict(torch.load("best_model.tar")) 

# predict dev
with open("predict_valid.txt", "w") as output:
    dev_correct = 0.0
    for words, tag in dev:
        # Padding (can be done in the conv layer as well)
        if len(words) < WIN_SIZE:
            words += [0] * (WIN_SIZE - len(words))
        words_tensor = torch.tensor(words).type(type)
        scores = model(words_tensor)[0]
        predict = scores.argmax().item()
        output.write(i2t[predict] + '\n')
        if predict == tag:
            dev_correct += 1
    dev_acc = dev_correct / len(dev)
    print("dev acc=%.4f" % (dev_acc))

# predict test
with open("predict_test.txt", "w") as output:
    for words, tag in test:
        # Padding (can be done in the conv layer as well)
        if len(words) < WIN_SIZE:
            words += [0] * (WIN_SIZE - len(words))
        words_tensor = torch.tensor(words).type(type)
        scores = model(words_tensor)[0]
        predict = scores.argmax().item()
        output.write(i2t[predict] + '\n')
