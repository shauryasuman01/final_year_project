import pennylane as qml
import torch
import torch.nn as nn

n_qubits = 4
n_qlayers = 1

dev = qml.device("default.qubit", wires=n_qubits)

def get_circuit():
    """
    Creates a variational quantum circuit for the QLSTM cell.
    """
    @qml.qnode(dev, interface='torch')
    def circuit(inputs, weights):
        # Angle Encoding of inputs
        # We assume inputs are normalized or scaled to a suitable range [0, PI]
        for i in range(n_qubits):
            # Use inputs[..., i] to handle both batched and unbatched inputs
            qml.RX(inputs[..., i], wires=i)
            qml.RY(inputs[..., i], wires=i)
            # We can repeat encoding if input dimension > n_qubits, but here we keep it simple
        
        # Variational Layer
        qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
        
        # Measurement
        return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]
    
    return circuit

class QLSTM(nn.Module):
    """
    Quantum LSTM Cell.
    Replaces the classical matrix multiplications with Quantum Variational Circuits.
    """
    def __init__(self, input_size, hidden_size):
        super(QLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_qubits = n_qubits
        
        # Ensure hidden_size matches n_qubits for direct mapping
        assert hidden_size == n_qubits, "Hidden size must match number of qubits for this simple VQC implementation"
        
        # We need circuits for Forget, Input, Update, and Output gates
        # Each circuit takes (input + hidden) as input. 
        # But VQC usually takes fixed inputs. 
        # We will use a classical linear layer to reduce (input+hidden) -> n_qubits
        # Then feed into VQC.
        
        self.cl_reduce = nn.Linear(input_size + hidden_size, n_qubits)
        
        self.qlayer_f = qml.qnn.TorchLayer(get_circuit(), {"weights": (n_qlayers, n_qubits)})
        self.qlayer_i = qml.qnn.TorchLayer(get_circuit(), {"weights": (n_qlayers, n_qubits)})
        self.qlayer_c = qml.qnn.TorchLayer(get_circuit(), {"weights": (n_qlayers, n_qubits)})
        self.qlayer_o = qml.qnn.TorchLayer(get_circuit(), {"weights": (n_qlayers, n_qubits)})

    def forward(self, x, init_states=None):
        batch_size, seq_len, _ = x.size()
        
        if init_states is None:
            h_t = torch.zeros(batch_size, self.hidden_size)
            c_t = torch.zeros(batch_size, self.hidden_size)
        else:
            h_t, c_t = init_states

        hidden_seq = []
        
        for t in range(seq_len):
            x_t = x[:, t, :]
            
            # Concatenate input and hidden state
            v_t = torch.cat((x_t, h_t), dim=1)
            
            # Reduce dimension classically to fit qubits
            # print(f"DEBUG: v_t shape: {v_t.shape}")
            try:
                v_reduced = torch.sigmoid(self.cl_reduce(v_t)) * 3.14 # Scaling to [0, pi]
            except Exception as e:
                print(f"Error in cl_reduce: {e}")
                print(f"v_t shape: {v_t.shape}")
                raise e
            
            # Quantum Gates
            try:
                f_t = torch.sigmoid(self.qlayer_f(v_reduced))
                i_t = torch.sigmoid(self.qlayer_i(v_reduced))
                # Candidate cell state
                g_t = torch.tanh(self.qlayer_c(v_reduced)) 
                o_t = torch.sigmoid(self.qlayer_o(v_reduced))
            except Exception as e:
                print(f"Error in Quantum Layers: {e}")
                print(f"v_reduced shape: {v_reduced.shape}")
                raise e
            
            # LSTM equations
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            
            hidden_seq.append(h_t.unsqueeze(1))
        
        hidden_seq = torch.cat(hidden_seq, dim=1)
        return hidden_seq, (h_t, c_t)

class QNLPHybridModel(nn.Module):
    """
    Hybrid Quantum-Classical Model for Sentiment Analysis.
    Structure: Embedding -> QLSTM -> Linear(1) -> Sigmoid
    """
    def __init__(self, vocab_size, embed_dim, hidden_size):
        super(QNLPHybridModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.qlstm = QLSTM(embed_dim, hidden_size)
        self.predict = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        embeds = self.embedding(x)
        lstm_out, _ = self.qlstm(embeds)
        # Take the last output for classification
        last_out = lstm_out[:, -1, :]
        logits = self.predict(last_out)
        return torch.sigmoid(logits)

# Test instantiation
if __name__ == "__main__":
    model = QNLPHybridModel(vocab_size=100, embed_dim=8, hidden_size=4)
    print("Model Architecture:")
    print(model)
