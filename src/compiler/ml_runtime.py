"""
Fourier ML Runtime Library
This code is injected into every compiled Fourier program that uses ML features.
"""

import torch
import torch.nn as nn

class FourierDense(nn.Module):
    """Dense (Fully Connected) Layer"""
    def __init__(self, in_features, out_features, activation=None):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation_name = activation
        
        # Map activation names to PyTorch functions
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "softmax":
            self.activation = nn.Softmax(dim=-1)
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            self.activation = None
    
    def forward(self, x):
        x = self.linear(x)
        if self.activation:
            x = self.activation(x)
        return x

class FourierSequential(nn.Module):
    """Sequential Model Container"""
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# Convenience constructors
def Dense(out_features, activation=None, in_features=None):
    """Create a Dense layer. in_features is auto-inferred if not provided."""
    if in_features is None:
        # Return a lambda that will be called with input size later
        return lambda in_feat: FourierDense(in_feat, out_features, activation)
    return FourierDense(in_features, out_features, activation)

def Sequential(layers):
    """Create a Sequential model from a list of layers."""
    return FourierSequential(layers)
