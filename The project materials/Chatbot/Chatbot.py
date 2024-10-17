
#loads all the contents from the libraries needed in this code

import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy 
import tflearn
import tensorflow as tf
import random
import json
import pickle

with open('dataset.py') as file: #opens the dataset and stores it in the variable name data
    data = json.load(file)

try:
   with open("data.pickle", "rb") as f:
       words, labels, training, output = pickle.load(f)

except:

    #blank lists to store all the unique words in our pattern
   
    words = [] 
    labels = []
    docs_x = []
    docs_y = []


    for intent in data["intents"]: #loops through the dataset file and extracts all the data from it
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern) #turns each patterns into a list of words
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    #stems the words to reduce the vocabulary of the model and find the more general meaning of the sentence and sorts it into a list

    words = [stemmer.stem(w.lower()) for w in words if w not in "?"] 
    words = sorted(list(set(words)))

    labels = sorted(labels)

    #represents each sentence with a list the length of the amount of words in our models vocabulary
    
    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)


        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = numpy.array(training) #converts the training data into numpy arrays
    output = numpy.array(output)

    with open("data.pickle", "wb") as f:
       pickle.dump((words, labels, training, output), f)

#creates the architecture of the model with neural networks

tf.compat.v1.reset_default_graph()

net = tflearn.input_data(shape= [None, len(training[0])]) 
net = tflearn.fully_connected(net,8)
net = tflearn.fully_connected(net,8)
net = tflearn.fully_connected(net,len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

try:
    model.load("model.tflearn")
except:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True) #trains the model on the data
    model.save("model.tflearn")
            
def bag_of_words(s, words): #converts the input from the user into a bag of words
    bag =[0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i]=(1)

    return numpy.array(bag)

def chat():
    print("Start talking with the bot!")
    print("Type quit to stop")
    while True:
        inp = input("You: ") #takes the inout from the user
        if inp.lower() == "quit":
            break

        results = model.predict([bag_of_words(inp,words)]) [0]
        results_index = numpy.argmax(results) #predicts the most probable class
        tag = labels[results_index]

        if results[results_index] > 0.7: 

            for tg in data["intents"]:
                if tg['tag'] == tag: #picks a response from that class
                    responses = tg['responses']

            print(random.choice(responses))
        else:
            print("I didn't get that, please try again.")
        
chat()
