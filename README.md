# Car Rental Chatbot

This project uses PyTorch as a deep learning framework to take a pre-made "intents" json file to use as training data to train a chatbot to respond to relevant user responses. Trained via feed-forward neural network

## Provided files 

intents.json : consists of tags (classes), patterns to recognize, and a small set of responses for each tag

nltk_utils.py : utilizes pythons natural language toolkit to define functions for stemming, tokenization, and a bag of words function to create trainable data

model.py : creates a Neural Network class with the ReLu function in torch.nn with 2 hidden layers

train.py : organizes training data with intents file and uses nltk_utils functions along with utils in PyTorch

chat.py : code to create chatbot

app.py : utilizes Tkinter to create a interactable gui to visualize chatbot 
