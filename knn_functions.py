import numpy as np

def euclidean_distance(x, y):
  return np.sqrt(np.sum((x-y)**2))

def manhattan_distance(x, y):
  return np.sum(np.abs(x - y))

def predict_for_one(X_train, y_train, x_test, k=15):
  distances = []
  for i, elem in X_train.iterrows():
    dist = manhattan_distance(elem, x_test)
    distances.append((dist, y_train[i]))
  distances.sort(key=lambda x: x[0])
  nearest_labels = [label for dist, label in distances[:k]]
  max_occur = max(nearest_labels, key= nearest_labels.count)
  return max_occur

def predict(X_train, y_train, X_test, k=15):
  predictions = []
  for i, row in X_test.iterrows():
    predictions.append(predict_for_one(X_train, y_train, row, k))
  return np.array(predictions)

def custom_accuracy(predictions, labels):
  arr = (predictions == labels)
  return sum(arr) / len(predictions)