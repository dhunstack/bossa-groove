# Feedforward neural network to predict the drum velocities and offsets given the drum hits
# The dataset is present in the hits.pkl, velocities.pkl and offsets.pkl files
# Need to load the dataset and train the model
# The model is saved in the model.pkl file
# The model is a feedforward neural network with 1 hidden layer of 256 neurons
# The input layer has 279 neurons, binary encoding of 9 drum hits with 31 time steps
# The output layer has 558 neurons, 9 drum velocities and 9 drum offsets with 31 time steps
# The model is trained using the Adam optimizer and Mean Squared Error loss function
# The model is trained for 100 epochs with a batch size of 32

import numpy as np
import pickle
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load the dataset
hits = pickle.load(open('hits.pkl', 'rb'))
velocities = pickle.load(open('velocities.pkl', 'rb'))
offsets = pickle.load(open('offsets.pkl', 'rb'))

# Filter the dataset
hits_flat = [h.flatten() for h in hits if h.shape[0] == 31]
velocities_flat = [v.flatten() for v in velocities if v.shape[0] == 31]
offsets_flat = [o.flatten() for o in offsets if o.shape[0] == 31]

# Convert the dataset to numpy arrays
X = np.array(hits_flat)
y1 = np.array(velocities_flat)
y2 = np.array(offsets_flat)
y = np.hstack((velocities_flat, offsets_flat))

# Split data into test and train sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Check sizes of input and output vectors
print("Size of train features matrix: ",X_train.shape, ", Size of train output vector: ",y_train.shape)
print("Size of test features matrix: ",X_test.shape, ", Size of test output vector: ",y_test.shape)

# Create the model
model = Sequential()
model.add(Dense(512, input_dim=279, activation='relu'))
model.add(Dense(512, input_dim=512, activation='relu'))
model.add(Dense(558, activation='linear'))

# Compile the model
model.compile(optimizer='Adam', loss='mse')

# Train the model
summary = model.fit(X_train, y_train, batch_size = 50, epochs = 30, validation_split=0.2, verbose=1)

# Save the model
model.save('model.keras')

# Evaluate the model
score_test = model.evaluate(X_test, y_test, verbose = 0)
print('Test loss', score_test)

# Predict the output for all data
y_pred = model.predict(X)
velocities_pred = y_pred[:, :279]
offsets_pred = y_pred[:, 279:]

# Save the required data
pickle.dump(X.reshape((len(X), 31, 9)), open('hits_orig.pkl', 'wb'))
pickle.dump(y1.reshape((len(X), 31, 9)), open('velocities_orig.pkl', 'wb'))
pickle.dump(y2.reshape((len(X), 31, 9)), open('offsets_orig.pkl', 'wb'))
pickle.dump(velocities_pred.reshape((len(y_pred), 31, 9)), open('velocities_pred.pkl', 'wb'))
pickle.dump(offsets_pred.reshape((len(y_pred), 31, 9)), open('offsets_pred.pkl', 'wb'))

# summarize history for loss
plt.plot(summary.history['loss'])
plt.plot(summary.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()