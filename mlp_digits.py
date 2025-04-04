import random
import math
import matplotlib.pyplot as plt
from matplotlib import font_manager

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        # Initialize weights and biases
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Initialize weights with small random values
        self.weights_input_hidden = [[random.uniform(-0.5, 0.5) for _ in range(hidden_size)] for _ in range(input_size)]
        self.weights_hidden_output = [[random.uniform(-0.5, 0.5) for _ in range(output_size)] for _ in range(hidden_size)]
        
        # Initialize biases
        self.bias_hidden = [random.uniform(-0.5, 0.5) for _ in range(hidden_size)]
        self.bias_output = [random.uniform(-0.5, 0.5) for _ in range(output_size)]
        
        # For tracking errors
        self.errors = []
    
    def sigmoid(self, x):
        # Prevent overflow in exp
        if x < -700:  # Approximate cutoff to prevent underflow
            return 0
        elif x > 700:  # Approximate cutoff to prevent overflow
            return 1
        return 1 / (1 + math.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward(self, inputs):
        # Calculate hidden layer activations
        hidden_inputs = [0] * self.hidden_size
        for i in range(self.hidden_size):
            for j in range(self.input_size):
                hidden_inputs[i] += inputs[j] * self.weights_input_hidden[j][i]
            hidden_inputs[i] += self.bias_hidden[i]
        
        hidden_outputs = [self.sigmoid(x) for x in hidden_inputs]
        
        # Calculate output layer activations
        final_inputs = [0] * self.output_size
        for i in range(self.output_size):
            for j in range(self.hidden_size):
                final_inputs[i] += hidden_outputs[j] * self.weights_hidden_output[j][i]
            final_inputs[i] += self.bias_output[i]
        
        final_outputs = [self.sigmoid(x) for x in final_inputs]
        
        return hidden_outputs, final_outputs
    
    def backpropagation(self, inputs, targets, hidden_outputs, final_outputs):
        # Calculate output errors
        output_errors = [targets[i] - final_outputs[i] for i in range(self.output_size)]
        
        # Update weights between hidden and output layers
        for i in range(self.hidden_size):
            for j in range(self.output_size):
                delta = output_errors[j] * self.sigmoid_derivative(final_outputs[j])
                self.weights_hidden_output[i][j] += self.learning_rate * hidden_outputs[i] * delta
        
        # Update output biases
        for i in range(self.output_size):
            self.bias_output[i] += self.learning_rate * output_errors[i] * self.sigmoid_derivative(final_outputs[i])
        
        # Calculate hidden layer errors
        hidden_errors = [0] * self.hidden_size
        for i in range(self.hidden_size):
            for j in range(self.output_size):
                hidden_errors[i] += output_errors[j] * self.weights_hidden_output[i][j]
        
        # Update weights between input and hidden layers
        for i in range(self.input_size):
            for j in range(self.hidden_size):
                delta = hidden_errors[j] * self.sigmoid_derivative(hidden_outputs[j])
                self.weights_input_hidden[i][j] += self.learning_rate * inputs[i] * delta
        
        # Update hidden biases
        for i in range(self.hidden_size):
            self.bias_hidden[i] += self.learning_rate * hidden_errors[i] * self.sigmoid_derivative(hidden_outputs[i])
    
    def train(self, training_data, targets, epochs):
        for epoch in range(epochs):
            total_error = 0
            for i in range(len(training_data)):
                # Forward pass
                hidden_outputs, final_outputs = self.forward(training_data[i])
                
                # Calculate error
                error = sum([(targets[i][j] - final_outputs[j])**2 for j in range(self.output_size)]) / self.output_size
                total_error += error
                
                # Backward pass
                self.backpropagation(training_data[i], targets[i], hidden_outputs, final_outputs)
            
            # Track average error for this epoch
            avg_error = total_error / len(training_data)
            self.errors.append(avg_error)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Error = {avg_error:.6f}")
    
    def predict(self, inputs):
        _, outputs = self.forward(inputs)
        return outputs.index(max(outputs))


def generate_font_digit_data(grid_size=7):
    """
    Generate pixel data for digits 0-9 in 5 different "fonts"
    Returns a list of input vectors and their corresponding targets
    """
    digits = []
    targets = []
    
    # Font 1: Basic block font
    font1 = [
        # 0
        [
            [0, 1, 1, 1, 0],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [0, 1, 1, 1, 0]
        ],
        # 1
        [
            [0, 0, 1, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 1, 1, 1, 0]
        ],
        # 2
        [
            [0, 1, 1, 1, 0],
            [1, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0],
            [1, 1, 1, 1, 1]
        ],
        # 3
        [
            [0, 1, 1, 1, 0],
            [1, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 1, 1, 0],
            [0, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [0, 1, 1, 1, 0]
        ],
        # 4
        [
            [0, 0, 0, 1, 0],
            [0, 0, 1, 1, 0],
            [0, 1, 0, 1, 0],
            [1, 0, 0, 1, 0],
            [1, 1, 1, 1, 1],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0]
        ],
        # 5
        [
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0],
            [1, 1, 1, 1, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [0, 1, 1, 1, 0]
        ],
        # 6
        [
            [0, 1, 1, 1, 0],
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 1, 1, 1, 0],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [0, 1, 1, 1, 0]
        ],
        # 7
        [
            [1, 1, 1, 1, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0]
        ],
        # 8
        [
            [0, 1, 1, 1, 0],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [0, 1, 1, 1, 0],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [0, 1, 1, 1, 0]
        ],
        # 9
        [
            [0, 1, 1, 1, 0],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [0, 1, 1, 1, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],
            [0, 1, 1, 1, 0]
        ]
    ]
    
    # Font 2: Slanted font
    font2 = [
        # 0
        [
            [0, 1, 1, 0, 0],
            [1, 0, 0, 1, 0],
            [1, 0, 0, 1, 0],
            [1, 0, 0, 1, 0],
            [1, 0, 0, 1, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0]
        ],
        # 1
        [
            [0, 0, 1, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0]
        ],
        # 2
        [
            [0, 1, 1, 0, 0],
            [1, 0, 0, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0]
        ],
        # 3
        [
            [0, 1, 1, 0, 0],
            [1, 0, 0, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [1, 0, 0, 1, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0]
        ],
        # 4
        [
            [0, 0, 1, 1, 0],
            [0, 1, 0, 1, 0],
            [1, 0, 0, 1, 0],
            [1, 1, 1, 1, 1],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0]
        ],
        # 5
        [
            [1, 1, 1, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0],
            [1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0]
        ],
        # 6
        [
            [0, 1, 1, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0],
            [1, 0, 0, 1, 0],
            [1, 0, 0, 1, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0]
        ],
        # 7
        [
            [1, 1, 1, 1, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ],
        # 8
        [
            [0, 1, 1, 0, 0],
            [1, 0, 0, 1, 0],
            [0, 1, 1, 0, 0],
            [1, 0, 0, 1, 0],
            [1, 0, 0, 1, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0]
        ],
        # 9
        [
            [0, 1, 1, 0, 0],
            [1, 0, 0, 1, 0],
            [1, 0, 0, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 1, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0]
        ]
    ]
    
    # Font 3: Thin font
    font3 = [
        # 0
        [
            [0, 1, 1, 1, 0],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [0, 1, 1, 1, 0]
        ],
        # 1
        [
            [0, 0, 1, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 1, 1, 1, 0]
        ],
        # 2
        [
            [0, 1, 1, 1, 0],
            [1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1]
        ],
        # 3
        [
            [0, 1, 1, 1, 0],
            [1, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [0, 1, 1, 1, 0]
        ],
        # 4
        [
            [1, 0, 0, 1, 0],
            [1, 0, 0, 1, 0],
            [1, 0, 0, 1, 0],
            [1, 1, 1, 1, 1],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0]
        ],
        # 5
        [
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0],
            [1, 1, 1, 1, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [0, 1, 1, 1, 0]
        ],
        # 6
        [
            [0, 1, 1, 1, 0],
            [1, 0, 0, 0, 0],
            [1, 1, 1, 1, 0],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [0, 1, 1, 1, 0]
        ],
        # 7
        [
            [1, 1, 1, 1, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0]
        ],
        # 8
        [
            [0, 1, 1, 1, 0],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [0, 1, 1, 1, 0],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [0, 1, 1, 1, 0]
        ],
        # 9
        [
            [0, 1, 1, 1, 0],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [0, 1, 1, 1, 1],
            [0, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [0, 1, 1, 1, 0]
        ]
    ]
    
    # Font 4: Bold font
    font4 = [
        # 0
        [
            [0, 1, 1, 1, 0],
            [1, 1, 0, 1, 1],
            [1, 1, 0, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 0, 1, 1],
            [1, 1, 0, 1, 1],
            [0, 1, 1, 1, 0]
        ],
        # 1
        [
            [0, 0, 1, 0, 0],
            [0, 1, 1, 0, 0],
            [1, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [1, 1, 1, 1, 1]
        ],
        # 2
        [
            [0, 1, 1, 1, 0],
            [1, 1, 0, 1, 1],
            [0, 0, 0, 1, 1],
            [0, 0, 1, 1, 0],
            [0, 1, 1, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1]
        ],
        # 3
        [
            [0, 1, 1, 1, 0],
            [1, 1, 0, 1, 1],
            [0, 0, 0, 1, 1],
            [0, 0, 1, 1, 0],
            [0, 0, 0, 1, 1],
            [1, 1, 0, 1, 1],
            [0, 1, 1, 1, 0]
        ],
        # 4
        [
            [0, 0, 1, 1, 0],
            [0, 1, 0, 1, 0],
            [1, 0, 0, 1, 0],
            [1, 1, 1, 1, 1],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0]
        ],
        # 5
        [
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0],
            [1, 1, 1, 1, 0],
            [0, 0, 0, 1, 1],
            [0, 0, 0, 0, 1],
            [1, 0, 0, 1, 1],
            [0, 1, 1, 1, 0]
        ],
        # 6
        [
            [0, 1, 1, 1, 0],
            [1, 1, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 1, 1, 1, 0],
            [1, 1, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [0, 1, 1, 1, 0]
        ],
        # 7
        [
            [1, 1, 1, 1, 1],
            [0, 0, 0, 1, 1],
            [0, 0, 1, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0]
        ],
        # 8
        [
            [0, 1, 1, 1, 0],
            [1, 1, 0, 1, 1],
            [1, 1, 0, 1, 1],
            [0, 1, 1, 1, 0],
            [1, 1, 0, 1, 1],
            [1, 1, 0, 1, 1],
            [0, 1, 1, 1, 0]
        ],
        # 9
        [
            [0, 1, 1, 1, 0],
            [1, 1, 0, 1, 1],
            [1, 0, 0, 0, 1],
            [0, 1, 1, 1, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 1],
            [0, 1, 1, 1, 0]
        ]
    ]
    
    # Font 5: Digital/LCD style font
    font5 = [
        # 0
        [
            [0, 1, 1, 1, 0],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 1, 1],
            [1, 0, 1, 0, 1],
            [1, 1, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [0, 1, 1, 1, 0]
        ],
        # 1
        [
            [0, 0, 1, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [1, 1, 1, 1, 1]
        ],
        # 2
        [
            [0, 1, 1, 1, 0],
            [1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1]
        ],
        # 3
        [
            [0, 1, 1, 1, 0],
            [1, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [0, 1, 1, 1, 0]
        ],
        # 4
        [
            [0, 0, 1, 1, 0],
            [0, 1, 0, 1, 0],
            [1, 0, 0, 1, 0],
            [1, 1, 1, 1, 1],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0]
        ],
        # 5
        [
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0],
            [1, 1, 1, 1, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [0, 1, 1, 1, 0]
        ],
        # 6
        [
            [0, 0, 1, 1, 0],
            [0, 1, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 1, 1, 1, 0],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [0, 1, 1, 1, 0]
        ],
        # 7
        [
            [1, 1, 1, 1, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0]
        ],
        # 8
        [
            [0, 1, 1, 1, 0],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [0, 1, 1, 1, 0],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [0, 1, 1, 1, 0]
        ],
        # 9
        [
            [0, 1, 1, 1, 0],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [0, 1, 1, 1, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0],
            [0, 1, 1, 0, 0]
        ]
    ]
    
    # Combine all fonts
    fonts = [font1, font2, font3, font4, font5]
    
    # Generate training data
    for font_idx, font in enumerate(fonts):
        for digit_idx, digit in enumerate(font):
            # Flatten the 2D grid to a 1D input array
            input_data = []
            for row in digit:
                input_data.extend(row)

    # Combine all fonts
    fonts = [font1, font2, font3, font4, font5]
    
    # Generate training data
    for font_idx, font in enumerate(fonts):
        for digit_idx, digit in enumerate(font):
            # Flatten the 2D grid to a 1D input array
            input_data = []
            for row in digit:
                input_data.extend(row)
            
            # Create one-hot encoded target
            target = [0] * 10
            target[digit_idx] = 1
            
            digits.append(input_data)
            targets.append(target)
    
    return digits, targets

def add_noise(data, noise_level=0.1):
    """
    Add random noise to the digit data
    noise_level controls the probability of flipping a bit
    """
    noisy_data = []
    for digit in data:
        noisy_digit = []
        for pixel in digit:
            if random.random() < noise_level:
                # Flip the bit (0->1 or 1->0)
                noisy_digit.append(1 - pixel)
            else:
                noisy_digit.append(pixel)
        noisy_data.append(noisy_digit)
    return noisy_data

def visualize_digit(digit, grid_size=5, title="Digit"):
    """
    Visualize a digit as a grid
    """
    # Reshape the 1D array to a 7x5 grid
    grid = []
    idx = 0
    for i in range(7):
        row = []
        for j in range(grid_size):
            row.append(digit[idx])
            idx += 1
        grid.append(row)
    
    # Create a figure
    plt.figure(figsize=(3, 3))
    plt.imshow(grid, cmap='binary')
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    
def plot_learning_curve(errors):
    """
    Plot the learning curve (error vs. epoch)
    """
    plt.figure(figsize=(10, 6))
    plt.plot(errors)
    plt.title('Learning Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.grid(True)
    plt.tight_layout()

def main():
    # Generate training data
    print("Generating training data...")
    training_data, targets = generate_font_digit_data()
    
    # Create a neural network with 35 input neurons (7x5 grid), 
    # 20 hidden neurons, and 10 output neurons (one for each digit)
    print("Creating neural network...")
    nn = NeuralNetwork(input_size=35, hidden_size=20, output_size=10, learning_rate=0.2)
    
    # Train the network
    print("Training neural network...")
    nn.train(training_data, targets, epochs=1000)
    
    # Plot the learning curve
    plot_learning_curve(nn.errors)
    plt.savefig('learning_curve.png')
    
    # Generate some test cases with noise
    print("Testing with noisy data...")
    test_data = add_noise(training_data, noise_level=0.1)
    
    # Display some test results
    correct = 0
    plt.figure(figsize=(15, 10))
    
    for i in range(10):
        # Choose a random digit
        idx = random.randint(0, len(test_data) - 1)
        
        # Make a prediction
        prediction = nn.predict(test_data[idx])
        actual = targets[idx].index(max(targets[idx]))
        
        # Keep track of accuracy
        if prediction == actual:
            correct += 1
        
        # Visualize the original and noisy digits
        plt.subplot(5, 4, i*2 + 1)
        visualize_digit(training_data[idx], title=f"Original: {actual}")
        
        plt.subplot(5, 4, i*2 + 2)
        visualize_digit(test_data[idx], title=f"Noisy: Pred={prediction}")
    
    plt.tight_layout()
    plt.savefig('test_results.png')
    
    print(f"Test accuracy: {correct / 10 * 100:.2f}%")
    
    # Interactive testing
    while True:
        try:
            font_idx = int(input("Enter font index (0-4) or -1 to quit: "))
            if font_idx == -1:
                break
                
            digit_idx = int(input("Enter digit (0-9): "))
            
            if font_idx < 0 or font_idx > 4 or digit_idx < 0 or digit_idx > 9:
                print("Invalid input. Font should be 0-4 and digit should be 0-9.")
                continue
                
            # Get the corresponding index in the training data
            idx = font_idx * 10 + digit_idx
            
            # Add noise at different levels
            noise_levels = [0.0, 0.1, 0.2, 0.3]
            plt.figure(figsize=(12, 3))
            
            for i, noise in enumerate(noise_levels):
                noisy_digit = add_noise([training_data[idx]], noise)[0]
                prediction = nn.predict(noisy_digit)
                
                plt.subplot(1, 4, i+1)
                visualize_digit(noisy_digit, title=f"Noise: {noise}\nPred: {prediction}")
            
            plt.tight_layout()
            plt.show()
            
        except ValueError:
            print("Please enter a valid number.")
        except KeyboardInterrupt:
            break
            
    print("Done!")

if __name__ == "__main__":
    main()