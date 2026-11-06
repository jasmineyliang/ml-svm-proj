import sys
sys.path.append('libsvm/python/')
from svmutil import svm_train, svm_predict
'''
Homework4: support vector machine classifier

You need to use two functions 'svm_train' and 'svm_predict'
from libsvm library to start your homework. Please read the
readme.txt file carefully to understand how to use these
two functions.

'''

def svm_with_diff_c(train_label, train_data, test_label, test_data):
	'''
	Use 'svm_train' function with training label, data and different value
	of cost c to train a svm classify model. Then apply the trained model
	on testing label and data.

	The value of cost c you need to try is listing as follow:
	c = [0.01, 0.1, 1, 2, 3, 5]
	Please keep other parameter options as default.
	No return value is needed
	'''
	c = [0.01, 0.1, 1, 2, 3, 5]
	for c_val in c:
		model = svm_train(train_label, train_data, f'-c {c_val} -q')
		p_label, p_acc, _ = svm_predict(test_label, test_data, model, '-q')
		print(f'C={c_val}: accuracy={p_acc[0]:.2f}%')

def svm_with_diff_kernel(train_label, train_data, test_label, test_data):
	'''
	Use 'svm_train' function with training label, data and different kernel
	to train a svm classify model. Then apply the trained model
	on testing label and data.

	The kernel  you need to try is listing as follow:
	1. linear kernel
	2. polynomial kernel
	3. radial basis function kernel
	Please keep other parameter options as default.
	No return value is needed
	'''
	kernels = [('-t 0', 'linear'), ('-t 1', 'poly'), ('-t 2', 'rbf')]
	for opts, name in kernels:
		model = svm_train(train_label, train_data, f'{opts} -q')
		p_label, p_acc, _ = svm_predict(test_label, test_data, model, '-q')
		print(f'kernel={name}: accuracy={p_acc[0]:.2f}%')
