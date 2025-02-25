import numpy as np
import pandas as pd

train_data = pd.read_csv('../data/mnist_train.csv')
test_data = pd.read_csv('../data/mnist_test.csv')

print("Original train and test data shapes:")
print(train_data.shape)
print(test_data.shape)


# Get subset of 2000 random training samples and 500 random test samples
train_data = train_data.sample(2000)
test_data = test_data.sample(500)

print("Subset train and test data shapes:")
print(train_data.shape)
print(test_data.shape)

# Save the subset data to csv files
train_data.to_csv('../data/mnist_train_subset.csv', index=False)
test_data.to_csv('../data/mnist_test_subset.csv', index=False)

print("Subset train and test data saved to csv files")
