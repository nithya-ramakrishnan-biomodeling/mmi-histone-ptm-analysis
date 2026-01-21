import numpy as np

# Create a 1D array
arr = np.array([1, 2, 3, 4, 5, 6])
print(f"Original Array: {arr}")

# Shuffle the array in-place
np.random.shuffle(arr)
print(f"Shuffled Array (in-place): {arr}")
