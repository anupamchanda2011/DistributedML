from flask import Flask,request
from NeuralNetwork import NeuralNetwork
import requests
import csv,pandas
import utils as utils

app = Flask(__name__)

n_classes = 0
def updateWeight(model,y,_y,eta=0.1):
	delta = (y - y_)
	# print("delta",delta)
	for layer in (model.network):
		for node in layer:
			arr = []
			for weights in node['weights']:
				# print(weights,eta,delta)
				weights = weights - eta*delta
				arr.append(weights)
			node['weights'] = arr
			# print(node)
	return model

def update_weight(network,x,y):
	global n_classes
	model = updateModel(network)
	# print("output_dim",n_classes)
	yhot_ = model._one_hot_encoding(y, n_classes) # one-hot target
	model._backward_pass(yhot_) # backward pass error (update node["delta"])
	network = model._update_weights(x, eta = 0.1)
	return network

def updateModel(network):
	eta = 0.1
	hidden_layers = [1,1]
	seed_weights = 1
	model = NeuralNetwork(hidden_layers=hidden_layers,seed=seed_weights,network=network)
	return model

@app.route('/compute',methods=['POST'])
def compute():
	network = request.json.get('network', "")
	x = request.json.get('x', "")
	network,x = initiate(network,x)
	# print(network)
	# data = {"network" : network}
	# requests.post('http://localhost:5040/compute',json = data)
	return "OK!"

def initiate():
	global y, n_classes
	filename = 'Iris.csv'
	eta = 0.1
	hidden_layers = [1,1]
	seed_weights = 1
	X, y, n_classes = utils.read_csv(filename, target_name="Species", normalize=False)
	# print("here",X[0])
	N, d = X.shape
	model = NeuralNetwork(input_dim=d, output_dim=n_classes,hidden_layers=hidden_layers, seed=seed_weights)
	return model,X,y
	

if __name__ == '__main__':
	#app.run(host='0.0.0.0', port=5050,debug = True)
	model,X,y = initiate()

	for layer in model.network:
		print(len(layer))
	# print(X,y)
	index = 0
	for x in X:
		p = x
		for i in range(len(model.network)): 
			network,x = model._forward_pass(x,layer_no = i)
		# y_ = network[len(model.network)-1][0]['output']
		# model = updateWeight(model,y[index],y_)
		network = update_weight(network,p,y[index])
		index = index+1
		# print(model.network[len(model.network)-1])
		# print(network)
		model = NeuralNetwork(input_dim = model.input_dim,output_dim = model.output_dim,hidden_layers=[1,1],seed=1,network=network)
		# model = NeuralNetwork(hidden_layers=[1],seed=1,network=model.network)
	print(model.network[len(model.network)-1])
	print(index)
	# print("here",model.network)
	# print(x)