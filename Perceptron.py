import numpy as np


class SingleLayerPerceptron:
    def __init__(self, num_inputs, learning_rate=0.1, epochs=1000):
        self.num_inputs = num_inputs
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = np.zeros(num_inputs + 1)  # Include weight for bias,Initialize weights to zero, including an extra weight for the bias term (hence num_inputs + 1).

    def forward_pass(self, inputs):
        # Compute the weighted sum (including bias)
        weighted_sum = np.dot(inputs, self.weights[1:]) + self.weights[0]
        # Apply the activation function (step function)
        return 1.0 if weighted_sum >= 0.0 else 0.0

    def train(self, inputs, labels):
        for epoch in range(self.epochs):
            total_error = 0
            for x, y in zip(inputs, labels):
                x_with_bias = np.insert(x, 0, 1.0)  # Add bias term (1.0) to the input
                y_pred = self.forward_pass(x_with_bias)
                error = y - y_pred
                total_error += abs(error)
                # Update weights
                self.weights += self.learning_rate * error * x_with_bias
            if epoch % 100 == 0:  # Print error every 100 epochs
                print(f"Epoch {epoch}, Total Error: {total_error}")
        return self.weights

    def predict(self, input_data):
        input_with_bias = np.insert(input_data, 0, 1.0)  # Add bias term (1.0) to the input
        return self.forward_pass(input_with_bias)

if __name__ == "__main__":
    # Training data for OR operation
    train_data = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]])
    labels_OR = np.array([1.0, 1.0, 1.0, 0.0])

    # Initialize the Perceptron
    num_features = train_data.shape[1]
    perceptron = SingleLayerPerceptron(num_inputs=num_features)

    # Train the Perceptron
    trained_weights = perceptron.train(train_data, labels_OR)

    # Test the Perceptron
    test_inputs = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]])
    print("\nTest Results:")
    for test_input in test_inputs:
        prediction = perceptron.predict(test_input)
        print(f"Input: {test_input} -> Prediction: {prediction}")

    print(f"\nTrained Weights: {trained_weights}")
