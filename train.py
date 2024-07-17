### file for training data
import json
from nltk_utils import tokenize, stem, bag_of_words
import numpy as np

#pytorch imports
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# import neural network
from model import NeuralNet 

## open and load json intents file
with open('intents.json', 'r') as f:
    intents = json.load(f)


## print(intents)

# create empty lists for training data
all_words = []
tags = []
xy = []
# loop through intents 
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    # loop through patterns
    for pattern in intent['patterns']:
        word = tokenize(pattern)
        all_words.extend(word)
        xy.append((word, tag)) #knows pattern and its corresponding tags

# punctuation to ignore
words_to_ignore = ['?', '!', '.', ',']
all_words = [stem(word) for word in all_words if word not in words_to_ignore]
#remove duplicated words and tags (in case)
all_words = sorted(set(all_words))
tags = sorted(set(tags))
#print(all_words)
#print(tags)

#tags
X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag) #X trained data is bag of words
    label = tags.index(tag)
    y_train.append(label) #CrossEntropyLoss 

# turn data into numpy arrays
X_train = np.array(X_train)
y_train = np.array(y_train)


#class for the Dataset of training data
class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train) 
        self.x_data = X_train
        self.y_data = y_train

    #dataset[index]
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index] # return tuple of xy data
    
    def __len__(self):
        return self.n_samples


#Hyperparameters
batch_size = 9 
input_size = len(all_words) #len of each bag_of_words = len all_words
hidden_size = 9
output_size = len(tags) #num_classes
learning_rate = 0.001
num_epochs = 1000

            ## testing if parameters are accurate
            #print(input_size, len(all_words))
            #print(output_size, tags)


# DataLoader allows us to batch train our data as it is now iterable
dataset = ChatDataset()
train_loader = DataLoader(dataset = dataset, batch_size = batch_size, shuffle=True, num_workers = 0)
# training model
    ## paremeters are hyperparameters
model = NeuralNet(input_size, hidden_size, output_size).to('cpu')

#loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        #push tuple to device
        words = words.to('cpu')
        labels = labels.to('cpu')

        #forward pass to next layer
        outputs = model(words)
        loss = criterion(outputs, labels)
        #backward and optimizer step
        optimizer.zero_grad() 
        loss.backward() # back propagation
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'epoch {epoch+1}/{num_epochs}, loss = {loss.item():.4f}') #print loss per 100 epochs

print(f'final loss = {loss.item():.4f}') ##print final loss

# save data at end of training process
data = {
    "model_state" : model.state_dict(),
    "input_size" : input_size,
    "output_size" : output_size,
    "hidden_size" : hidden_size,
    "all_words" : all_words,
    "tags" : tags
}
FILE = "data.pth"
torch.save(data, FILE)
print(f'training complete. file saved to {FILE}')