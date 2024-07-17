#chatbot file

from lib2to3.pgen2 import token
import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize, stem

with open('intents.json', 'r') as f:
    intents = json.load(f)


FILE = "data.pth"
data = torch.load(FILE)

# get data from file to store as parameters
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to('cpu')
model.load_state_dict(model_state)
model.eval() ##evaluation mode


### CHATBOT code

bot_name = "CHARLES"

def get_response(input_msg):
    # tokenize input & bog
    sentence_input = tokenize(input_msg)
    X_data = bag_of_words(sentence_input, all_words)
    X_data = X_data.reshape(1,X_data.shape[0]) #give it 1 row
    X_data = torch.from_numpy(X_data)
    # get predicted output
    output = model(X_data)
    _, predicted = torch.max(output, dim = 1)
    tag = tags[predicted.item()] #class label at index of predicted output

    # check probability of tag to see if it is large enough
    probabilities = torch.softmax(output, dim = 1)
    prob = probabilities[0][predicted.item()]

    if prob.item() > 0.75:
        # loop over all intents to check if the tag matches
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                # select random response from corresponding tag class
                return random.choice(intent['responses'])

    return "I do not understand ..."

print(f"{bot_name}: Hello, how may I help you today? (enter 'exit' to exit the chat) ")












