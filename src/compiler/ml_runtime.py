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
    """Sequential Model Container with Lazy Building"""
    def __init__(self, layers_data):
        super().__init__()
        self.raw_layers = layers_data
        self.built_layers = nn.ModuleList()
        self.initialized = False
    
    def _build(self, input_data):
        current_size = input_data.shape[-1]
        for layer_item in self.raw_layers:
            if callable(layer_item) and not isinstance(layer_item, nn.Module):
                # It's a constructor lambda: lambda in_feat: FourierDense(...)
                built_layer = layer_item(current_size)
            else:
                built_layer = layer_item
            
            self.built_layers.append(built_layer)
            # Update current_size for next layer
            if hasattr(built_layer, 'linear'):
                current_size = built_layer.linear.out_features
            elif isinstance(built_layer, nn.Linear):
                current_size = built_layer.out_features
        self.initialized = True

    def forward(self, x):
        if not self.initialized:
            self._build(x)
        for layer in self.built_layers:
            x = layer(x)
        return x

# Convenience constructors
def Dense(out_features, activation=None, in_features=None):
    """Create a Dense layer. in_features is auto-inferred if not provided."""
    if in_features is None:
        return lambda in_feat: FourierDense(in_feat, out_features, activation)
    return FourierDense(in_features, out_features, activation)

def Sequential(layers):
    """Create a Sequential model from a list of layers."""
    return FourierSequential(layers)
