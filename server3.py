from flask import Flask,request
from NeuralNetwork import NeuralNetwork
import requests
import csv,pandas
import utils as utils
import numpy as np

app = Flask(__name__)

count = 0
yhot_ = []
layer_index = 0
servers = []
params = {}

def back_prop(network,yhot_):
	global n_classes,layer_index
	model = updateModel(network)
	yhot = np.array(yhot_)
	network = model._backward_pass_for_layer(yhot = yhot,layer = layer_index) # backward pass error (update node["delta"])
	return network

def updateModel(network):
	global params
	eta = params['eta']
	hidden_layers = params['hidden_layers']
	seed_weights = params['seed_weight']
	model = NeuralNetwork(hidden_layers=hidden_layers,seed=seed_weights,network=network)
	return model

@app.route('/backprop',methods=['POST'])
def backprop():
	global yhot_,servers
	network = request.json.get('network', "")

	model = updateModel(network)
	network = back_prop(network,yhot_)

	### SENDING MODEL BACK TO SERVER 2 AFTER Back Prop
	data = {"network" : network}
	addr = servers[layer_index-1] + '/backprop'
	requests.post(addr,json = data)
	return "OK!"

@app.route('/compute',methods=['POST'])
def compute():
	global yhot_,layer_index,servers,params
	network = request.json.get('network', "")
	x = request.json.get('x', "")
	yhot_ = request.json.get('yhot',"")
	layer_index = request.json.get('layer_index', "")
	layers = request.json.get('layers',"")
	servers = request.json.get('servers',"")
	params = request.json.get('params',"")

	layer_index = int(layer_index) + 1
	model = updateModel(network)
	network,x = model._forward_pass(x,layer_no = layer_index)
	
	### BACK PROPAGATION STARTS HERE
	if layer_index == int(layers)-1:
		network = back_prop(network,yhot_)
		data = {"network" : network}
		addr = servers[layer_index-1] + '/backprop'
		requests.post(addr,json = data)
		return "OK!"
	else:
	### CONTINUING SENDING MODEL PARAMS TO NEXT LAYER 	
		data = {"network" : network, "x" : x,"yhot" : yhot_,"layer_index": layer_index,
		"layers" : layers,"servers" : servers,"params" : params}
		addr = servers[layer_index+1] + "/compute"
		requests.post(addr,json = data)
		return "OK!"
	
if __name__ == '__main__':
	app.run(host='0.0.0.0', port=5040,debug = True)