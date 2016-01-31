import pickle
import sys

layers, _, schedule = pickle.load(sys.argv[1])

print layers
print schedule
