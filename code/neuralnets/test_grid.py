# from neuralnets import mnist_loader, network, plot
# import sys
# import os
# sys.path.insert(0, os.path.abspath('..'))
# from neuralnets import network_grid
# import network, mnist_loader
import network, network_grid, mnist_loader
import numpy as np
import time


def test_grid():
    training_data, _, test_data = mnist_loader.load_data_wrapper()
    net = network_grid.Network([28*28, 30, 10])
    net2 = network.Network([28*28, 30, 10])
    start_time = time.time()
    net.SGD(training_data=training_data, epochs=10, mini_batch_size=15, eta=0.1, test_data=test_data)
    print("--- %s seconds ---" % (time.time() - start_time))
    training_data, _, test_data = mnist_loader.load_data_wrapper()
    start_time = time.time()
    net2.SGD(training_data=training_data, epochs=10, mini_batch_size=15, eta=0.1, test_data=test_data)
    print("--- %s seconds ---" % (time.time() - start_time))




test_grid()

# def vectorized_result(j, l):
#     """Return a 10-dimensional unit vector with a 1.0 in the jth
#     position and zeroes elsewhere.  This is used to convert a digit
#     (0...9) into a corresponding desired output from the neural
#     network."""
#     e = np.zeros((l, 1))
#     e[j] = 1.0
#     return e
# def add_two_numbers():
#     td_size = 100
#     num_arrays = [np.random.choice(range(10), replace=True, size=(2, 1)) for x in range(td_size)]
#     # Training data has the shape [(2,1),(19,1)], as the second item is the sum represented in array form
#     training_data = [(num_array, vectorized_result(j=np.sum(num_array), l=19)) for num_array in num_arrays[:td_size-100]]
#     # Note how we are not vectoring but just return the summation
#     # Test data has the shape (2,1), 1, as the second item is just the number 0-18 of the sum
#     test_data = [(num_array, np.sum(num_array)) for num_array in num_arrays[td_size-100:]]
#     net = network_grid.Network([2, 19, 19])
#     print("training network")

#     print("correct results", net.SGD(training_data=training_data, test_data=test_data,eta=1, epochs=100, mini_batch_size=5, should_print=False, mean=True))