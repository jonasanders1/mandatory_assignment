import pandas as pd
 
 
"""
I created a subset of the MNIST dataset and deleted the original files due to git lfs size limits.
""" 
train_data = pd.read_csv('../data/mnist_train.csv')
test_data = pd.read_csv('../data/mnist_test.csv')
 
train_data = train_data.sample(n=2000)
test_data = test_data.sample(n=500)

print(train_data.shape)
print(test_data.shape)

train_data.to_csv('../data/mnist_train_small.csv', index=False)
test_data.to_csv('../data/mnist_test_small.csv', index=False)

 
 
 
 