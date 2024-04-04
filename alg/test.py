import pickle
with open('data.pickle', 'rb') as handle:
    unserialized_data = pickle.load(handle)
print(unserialized_data)