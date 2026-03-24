import pickle

data = pickle.load(open('accuracy_pdb.pkl', 'rb'))

print(len(data[0]))
print(len((data[0])))

print(data[0][0])