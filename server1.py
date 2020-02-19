from flask import Flask,request
from NeuralNetwork import NeuralNetwork
import requests
import csv,pandas
import utils as utils
import numpy as np
import json

app = Flask(__name__)

data_index = 0
n_epochs = 0
X = []
y = []
model = {}
n_classes = 0
count = 0
yhot_ = []
layer_index = 0
layers = 0
servers = []
params = {}

def update_weight(network,x):
	global n_classes, count, yhot_,layer_index
	count +=1
	model = updateModel(network)
	yhot = np.array(yhot_)
	network = model._backward_pass_for_layer(yhot,layer = layer_index) # backward pass error (update node["delta"])
	model = updateModel(network)
	network = model._update_weights(x, eta = 0.1)
	return network

@app.route('/backprop',methods=['POST'])
def sending():
	global X, data_index, n_epochs, y,count, yhot_,layer_index,layers,servers,params
	network = request.json.get('network', "")
	network = update_weight(network,X[data_index-1])
	if data_index < len(X):
		model = updateModel(network)
		network,x = model._forward_pass(X[data_index],layer_no = layer_index)
		yhot_ = model._one_hot_encoding(y[data_index], n_classes)
		data_index = data_index + 1;
		data = {"network" : network,"x" : x,"yhot" : yhot_.tolist(),
		"layer_index": layer_index,"layers" : layers,"servers" : servers,"params" : params}
		addr = servers[layer_index+1] + "/compute"
		requests.post(addr,json = data)
		return "OK!" 
	else:
		print(network[len(network) - 1])
		print(count)
		return "OK"

@app.route('/')
def compute():
	# Global Variables
	global X, data_index, yhot_,layer_index,layers,servers,params
	data_index = 0
	model,X,Y,layers = initiate()
	# X = X[0:100]
	layer_index = 0
	network,x = model._forward_pass(X[0],layer_no = layer_index)
	yhot_ = model._one_hot_encoding(Y[0], n_classes)
	data_index += 1
	data = {"network" : network,"x" : x,"yhot" : yhot_.tolist(),"layer_index": layer_index,
	"layers" : layers,"servers" : servers,"params" : params}
	addr = servers[layer_index+1] + "/compute"
	# requests.post('http://localhost:5030/compute',json = data)
	requests.post(addr,json = data)
	return "OK!"

def updateModel(network):
	global params
	eta = params['eta']
	hidden_layers = params['hidden_layers']
	seed_weights = params['seed_weight']
	model = NeuralNetwork(hidden_layers=hidden_layers,seed=seed_weights,network=network)
	return model

def initiate():
	global y,n_classes,servers,params
	# Reading from config file ###
	with open('./config.json') as f:
		data = json.load(f)
	filename = data['filename']
	servers = data['servers']
	params = data['params']
	eta = params['eta']
	hidden_layers = params['hidden_layers']
	seed_weights = params['seed_weight']

	layers = len(hidden_layers) + 1
	X, y, n_classes = utils.read_csv(filename, target_name="Species", normalize=False)
	N, d = X.shape
	model = NeuralNetwork(input_dim=d, output_dim=n_classes,hidden_layers=hidden_layers, seed=seed_weights)
	return model,X,y,layers
			
if __name__ == '__main__':
	app.run(host='0.0.0.0', port=5020,debug = True)

