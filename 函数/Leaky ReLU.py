import numpy as np

def leaky_relu(x, alpha=0.01):
    """Leaky ReLU activation function with slope alpha."""
    return np.maximum(alpha * x, x)

# Example usage
x = np.array([-1.0, 2.0, 3.0, -4.0])
y = leaky_relu(x)
print(y)  # Output: [-0.01  2.    3.   -0.04]
