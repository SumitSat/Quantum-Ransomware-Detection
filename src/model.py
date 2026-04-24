import torch
import torch.nn as nn
import pennylane as qml

class ClassicalEncoder(nn.Module):
    """
    Standard Feed-Forward Neural Network to reduce 
    high-dimensional static PE features down to the number of qubits.
    """
    def __init__(self, input_dim, output_dim=8):
        super().__init__()
        # Reduce ~530 dimensions down to 64, then to 8 for the qubits
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, output_dim),
            nn.Tanh() # Tanh scales outputs to [-1, 1], which is great for quantum rotation angles
        )
        
    def forward(self, x):
        return self.net(x)

def create_vqc_layer(n_qubits=8, n_layers=2, device_string="default.qubit"):
    """
    Creates the PennyLane Quantum Node (QNode) and wraps it in a TorchLayer.
    """
    dev = qml.device(device_string, wires=n_qubits)

    @qml.qnode(dev, interface='torch')
    def quantum_circuit(inputs, weights):
        # AngleEmbedding natively handles PyTorch batch dimensions!
        qml.AngleEmbedding(inputs * torch.pi, wires=range(n_qubits), rotation='Y')
            
        # Variational layers (StronglyEntanglingLayers provides rotation and CNOTs)
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        
        # We take the expectation value of the first qubit as the binary decision
        return qml.expval(qml.PauliZ(0))

    weight_shapes = {"weights": (n_layers, n_qubits, 3)}
    vqc_layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)
    return vqc_layer

class HybridQuantumNet(nn.Module):
    """
    The Q-TERD Path A Architecture:
    High-Dim Static Features -> Classical Reduction -> Quantum Circuit -> Sigmoid
    """
    def __init__(self, input_dim, n_qubits=8, n_layers=2, device_string="default.qubit"):
        super().__init__()
        self.encoder = ClassicalEncoder(input_dim=input_dim, output_dim=n_qubits)
        self.vqc = create_vqc_layer(n_qubits, n_layers, device_string)
        self.fc = nn.Linear(1, 1) 
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        encoded = self.encoder(x)
        q_out = self.vqc(encoded)
        if len(q_out.shape) == 1:
            q_out = q_out.unsqueeze(1)
        return self.sigmoid(self.fc(q_out))

class StrictClassicalDNN(nn.Module):
    """
    Apples-to-Apples benchmark. Uses the exact same ClassicalEncoder
    but replaces the Quantum VQC with a standard classical dense layer.
    """
    def __init__(self, input_dim, hidden_dim=8):
        super().__init__()
        self.encoder = ClassicalEncoder(input_dim=input_dim, output_dim=hidden_dim)
        # Replacing the VQC + single node mapping with a classical linear mapping
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        encoded = self.encoder(x)
        out = self.fc(encoded)
        return self.sigmoid(out)
