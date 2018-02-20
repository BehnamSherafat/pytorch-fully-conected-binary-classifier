
"""
Binary classification by running a fully connected multi-layer perceptron model with two hidden units using pytorch:
"""

import torch
import numpy as np 
from torch.autograd import Variable
import torch.nn.functional as func


#To simulate data, let's fix number of inputs and observations:
n_train, n_test, p = 100000, 10000, 5

n = n_test + n_train

x = np.random.normal(0., 1., size = (n, p))

#Here we aim to simulate a matrix of features that contain p features, and square of them and interaction between all two-
#by-two original features. This will be hard for a simple regression (without incorporating squares and interactions)...

#number of features (p is the number of original inputs)
pp = p + p + int(p* (p - 1)/2)

xx = np.zeros((n, pp))

xx[:, :p] = x
k = p 

for i in range(p-1):
	temp = x[:, i]**2
	xx[:, k] = temp
	k += 1
	for j in range(i + 1, p):
		temp = x[:, i]*x[:, j]
		xx[:, k] = temp
		k += 1

temp = x[:, p-1]**2
xx[:, k] = temp


true_w = np.random.uniform(-3.5, 3.5, size = (pp, 1))

z = np.dot(xx, true_w) + np.random.normal(0., 1., size = (n, 1))
t = 2. * np.tanh(z/2.) + 1


#Dichotomizing the continuous variable for logit model.
t = np.where(t > 0., 1., 0.)


x_data = Variable(torch.from_numpy(xx[:n_train, :]).float(), requires_grad = False)
t_data = Variable(torch.from_numpy(t[:n_train]).float(), requires_grad = False)

#print(x_data.data.shape)


#Need to fix the architecture of our neural network:
hidden = [50, 30]

#Number of neurons in the last layer (output layer):
c = 1

class bc_mlp(torch.nn.Module):#binary classification class as a child of torch.nn.Module
	
	def __init__(self):
		super(bc_mlp, self).__init__()
		self.lin_lay1 = torch.nn.Linear(pp, hidden[0])
		self.lin_lay2 = torch.nn.Linear(hidden[0], hidden[1])
		self.lin_output = torch.nn.Linear(hidden[1], c)

	def forward(self, inputs):
		a_lay1 = func.relu(self.lin_lay1(inputs))
		a_lay2 = func.relu(self.lin_lay2(a_lay1))
		output = func.sigmoid(self.lin_output(a_lay2))
		return(output)

my_bc_mlp = bc_mlp()

loss = torch.nn.BCELoss(size_average = False)
optimizer = torch.optim.Adam(my_bc_mlp.parameters(), lr = .01)


for i in range(10):
	#forward, loss, backward
	y_pred = my_bc_mlp(x_data)

	l = loss(y_pred, t_data)

	optimizer.zero_grad()

	l.backward()

	optimizer.step()


x_test = Variable(torch.from_numpy(xx[n_train:n, :]).float(), requires_grad = False)
t_test = Variable(torch.from_numpy(t[n_train:n]).float(), requires_grad = False)


#Now we use the forward method we defined our model class.
results = my_bc_mlp.forward(x_test)

def accuracy(y, t, threshold = .5):
	"""y and t are tensors"""
	y = y.data.numpy()
	t = t.data.numpy()
	y_cat = 0. + (y >= threshold)
	a11 = np.dot(y_cat.T, t);
	a12 = np.dot(y_cat.T, (1. - t))
	a21 = np.dot((1. - y_cat).T, t)
	a22 = np.dot((1. - y_cat).T, (1. - t))
	print("Confusion matrix (predicted vs. observed):")
	confuse = np.vstack((np.hstack((a11, a12)), np.hstack((a21, a22))))
	print(confuse)
	print("Sensitivity (%):", np.round(100*a11/(a11 + a21), 1))
	print("Specificity (%):", np.round(100*a22/(a22 + a12), 1))
	print("Accuracy (%):", np.round(100*(a11 + a22)/np.sum(confuse), 1))

print(accuracy(results, t_test))
