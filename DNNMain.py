import numpy as np

from components.DNNNetwork import DynamicNeuralNetwork

if __name__ == '__main__':
    trial_network = DynamicNeuralNetwork([(1, 2), (3, 4)], [(2, 3)],
                                         input_node_connectivity=1.0, output_node_connectivity=1.0)
    input_a = np.random.random((1, 2))
    input_b = np.random.random((3, 4))
    expected_output = np.random.random((2, 3))

    trial_network.add_input_data([input_a, input_b])
    trial_network.propagate_inputs()
    outputs = trial_network.extract_output_data()
    print(outputs)
    trial_network.clear_incoming_messages()

    trial_network.perform_backpropagation([input_a, input_b], [expected_output])

    trial_network.add_input_data([input_a, input_b])
    trial_network.propagate_inputs()
    outputs = trial_network.extract_output_data()
    print(outputs)
    trial_network.clear_incoming_messages()
