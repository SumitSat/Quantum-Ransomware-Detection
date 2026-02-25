import torch
import torch.nn as nn
import pennylane as qml

class CNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(10, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.fc = nn.Linear(32 * 120, 64) # Output dimension 64
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # x: (Batch, Seq, Channels) -> (Batch, Channels, Seq)
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc(x))
        return x

def create_vqc_layer(n_qubits=8, n_layers=2, device_string="default.qubit"):
    """
    Creates the PennyLane Quantum Node (QNode) and wraps it in a TorchLayer.
    """
    dev = qml.device(device_string, wires=n_qubits)

    @qml.qnode(dev, interface='torch')
    def quantum_circuit(inputs, weights):
        # Angle Encoding: Map 16 features to 8 qubits
        for i in range(n_qubits):
            qml.RY(inputs[i], wires=i) # Features 0-7
            qml.RZ(inputs[i+n_qubits], wires=i) # Features 8-15
            
        # Variational layers
        qml.StrongEntanglingLayers(weights, wires=range(n_qubits))
        
        return qml.expval(qml.PauliZ(0))

    weight_shapes = {"weights": (n_layers, n_qubits, 3)}
    vqc_layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)
    return vqc_layer

class HybridQuantumNet(nn.Module):
    def __init__(self, n_qubits=8, n_layers=2, device_string="default.qubit"):
        super().__init__()
        self.vqc = create_vqc_layer(n_qubits, n_layers, device_string)
        self.fc = nn.Linear(1, 1) 
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x is 16-dim PCA output -> (Batch, 16)
        q_out = self.vqc(x)
        if len(q_out.shape) == 1:
            q_out = q_out.unsqueeze(1)
        return self.sigmoid(self.fc(q_out))
