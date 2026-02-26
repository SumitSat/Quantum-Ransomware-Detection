import torch
import torch.nn as nn
import pennylane as qml

class QLSTMCell(nn.Module):
    """
    Quantum-LSTM Cell.
    Replaces the standard classic dense layers in LSTM gates (forget, input, output, cell) 
    with Variational Quantum Circuits (VQCs).
    """
    def __init__(self, input_size, hidden_size, n_qubits=4, n_layers=1, backend="default.qubit"):
        super(QLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_qubits = n_qubits
        
        # Scaling layer: maps concated [x_t, h_{t-1}] to the number of qubits
        self.cl_dim_reduction = nn.Linear(input_size + hidden_size, n_qubits)
        
        # Define the Pennylane QNodes for the 4 LSTM gates
        self.dev = qml.device(backend, wires=n_qubits)
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        
        # We need 4 VQCs: forget(f), input(i), candidate/update(g), output(o)
        self.vqc_forget = qml.qnn.TorchLayer(self._make_qnode(), weight_shapes)
        self.vqc_input = qml.qnn.TorchLayer(self._make_qnode(), weight_shapes)
        self.vqc_update = qml.qnn.TorchLayer(self._make_qnode(), weight_shapes)
        self.vqc_output = qml.qnn.TorchLayer(self._make_qnode(), weight_shapes)
        
        # Dimension upscaling (n_qubits -> hidden_size)
        self.cl_out_f = nn.Linear(n_qubits, hidden_size)
        self.cl_out_i = nn.Linear(n_qubits, hidden_size)
        self.cl_out_g = nn.Linear(n_qubits, hidden_size)
        self.cl_out_o = nn.Linear(n_qubits, hidden_size)

    def _make_qnode(self):
        @qml.qnode(self.dev, interface='torch')
        def circuit(inputs, weights):
            # Encoding
            for i in range(self.n_qubits):
                qml.RY(inputs[i], wires=i)
            # Parametrized Layers
            qml.StrongEntanglingLayers(weights, wires=range(self.n_qubits))
            # Measurement
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        return circuit

    def forward(self, x_t, h_t, c_t):
        concat_input = torch.cat([x_t, h_t], dim=1)           # (batch, input_size + hidden_size)
        vqc_input = self.cl_dim_reduction(concat_input)       # (batch, n_qubits) -> Compress
        vqc_input = torch.tanh(vqc_input)                     # Normalize for angle embedding
        
        # VQC evaluations
        f_out = self.vqc_forget(vqc_input)
        i_out = self.vqc_input(vqc_input)
        g_out = self.vqc_update(vqc_input)
        o_out = self.vqc_output(vqc_input)
        
        # Projection and Activations
        f_t = torch.sigmoid(self.cl_out_f(f_out))
        i_t = torch.sigmoid(self.cl_out_i(i_out))
        g_t = torch.tanh(self.cl_out_g(g_out))
        o_t = torch.sigmoid(self.cl_out_o(o_out))
        
        # LSTM Cell State Update
        c_next = (f_t * c_t) + (i_t * g_t)
        h_next = o_t * torch.tanh(c_next)
        
        return h_next, c_next

class QTERD_QLSTM(nn.Module):
    """
    Main sequence modeling class for Quantum-Temporal Early Ransomware Detection.
    """
    def __init__(self, input_size, hidden_size, n_qubits=4, n_vqc_layers=1, backend="default.qubit"):
        super(QTERD_QLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.qlstm_cell = QLSTMCell(input_size, hidden_size, n_qubits, n_vqc_layers, backend)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        batch_size, seq_len, _ = x.size()
        
        # Initialize hidden and cell states
        h_t = torch.zeros(batch_size, self.hidden_size).to(x.device)
        c_t = torch.zeros(batch_size, self.hidden_size).to(x.device)
        
        # Step through time
        for t in range(seq_len):
            x_t = x[:, t, :]
            h_t, c_t = self.qlstm_cell(x_t, h_t, c_t)
            
        # Classify based on final hidden state
        out = self.fc(h_t)
        out = self.sigmoid(out)
        return out
