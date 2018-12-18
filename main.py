# download from http://nlp.stanford.edu/data/glove.twitter.27B.zip
# WORD_VECTORS = "../embeddings/glove.twitter.27B.50d.txt"
import os
import warnings

from sklearn.exceptions import UndefinedMetricWarning
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import DataLoader

from config import EMB_PATH
from dataloading import SentenceDataset
from models import DNN
from training import train_dataset, eval_dataset
from utils.load_datasets import load_MR
from utils.load_embeddings import load_word_vectors

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

########################################################
# Configuration
########################################################

EMBEDDINGS = os.path.join(EMB_PATH, "glove.6B.50d.txt")
EMB_DIM = 50
EMB_TRAINABLE = False
BATCH_SIZE = 128
EPOCHS = 50

# if your computer has a CUDA compatible gpu use it, otherwise use the cpu
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

########################################################
# Define PyTorch datasets and dataloaders
########################################################

# load word embeddings
print("loading word embeddings...")
word2idx, idx2word, embeddings = load_word_vectors(EMBEDDINGS, EMB_DIM)

# load the raw data
# X_train, y_train, X_test, y_test = load_Semeval2017A()
X_train, y_train, X_test, y_test = load_MR()

# convert labels from strings to integers
label_encoder = LabelEncoder()
label_encoder = label_encoder.fit(y_train)

y_train = label_encoder.transform(y_train)
y_test = label_encoder.transform(y_test)
n_classes = label_encoder.classes_.size

# Define our PyTorch-based Dataset
train_set = SentenceDataset(X_train, y_train, word2idx)
test_set = SentenceDataset(X_test, y_test, word2idx)

# Define our PyTorch-based DataLoader
train_loader = DataLoader(...)
test_loader = DataLoader(...)

#############################################################################
# Model Definition (Model, Loss Function, Optimizer)
#############################################################################
model = DNN(num_classes=n_classes,
            embeddings=embeddings,
            trainable_emb=EMB_TRAINABLE)

# move the mode weight to cpu or gpu
model.to(DEVICE)

print(model)

criterion = torch.nn.CrossEntropyLoss()

# We optimize ONLY those parameters that are trainable (p.requires_grad==True)
parameters = filter(lambda p: p.requires_grad, model.parameters())
optimizer = torch.optim.Adam(parameters)

#############################################################################
# Training Pipeline
#############################################################################
for epoch in range(1, EPOCHS + 1):
    # train the model for one epoch
    train_dataset(epoch, train_loader, model, criterion, optimizer)

    # evaluate the performance of the model, on both data sets
    ... = eval_dataset(train_loader, model, criterion)
    ... = eval_dataset(test_loader, model, criterion)
