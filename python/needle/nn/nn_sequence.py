"""The module."""

from typing import List
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.power_scalar(1 + ops.exp(-x), -1)
        ### END YOUR SOLUTION


class RNNCell(Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        bias=True,
        nonlinearity="tanh",
        device=None,
        dtype="float32",
    ):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        bound = 1 / np.sqrt(hidden_size)
        self.W_ih = Parameter(
            init.rand(
                input_size,
                hidden_size,
                low=-bound,
                high=bound,
                dtype=dtype,
                device=device,
            )
        )
        self.W_hh = Parameter(
            init.rand(
                hidden_size,
                hidden_size,
                low=-bound,
                high=bound,
                dtype=dtype,
                device=device,
            )
        )
        if bias is True:
            self.bias_ih = Parameter(
                init.rand(
                    hidden_size, 1, low=-bound, high=bound, dtype=dtype, device=device
                )
            )
            self.bias_hh = Parameter(
                init.rand(
                    hidden_size, 1, low=-bound, high=bound, dtype=dtype, device=device
                )
            )
        else:
            self.bias_ih = None
            self.bias_hh = None
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.nonlinearity = nonlinearity
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        if h is None:
            h = init.zeros(
                X.shape[0], self.W_hh.shape[0], device=X.device, dtype=X.dtype
            )

        ht = X @ self.W_ih + h @ self.W_hh
        if self.bias is True:
            broadcast_bih = ops.broadcast_to(
                ops.reshape(self.bias_ih, (1, -1)), (X.shape[0], self.hidden_size)
            )
            broadcast_bhh = ops.broadcast_to(
                ops.reshape(self.bias_hh, (1, -1)), (X.shape[0], self.hidden_size)
            )
            bias = broadcast_bih + broadcast_bhh
            ht += bias
        if self.nonlinearity == "tanh":
            ht = ops.tanh(ht)
        elif self.nonlinearity == "relu":
            ht = ops.relu(ht)
        else:
            raise ValueError
        return ht
        ### END YOUR SOLUTION


class RNN(Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
        bias=True,
        nonlinearity="tanh",
        device=None,
        dtype="float32",
    ):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.nonlinearity = nonlinearity
        self.bias = bias
        self.rnn_cells = []
        self.rnn_cells.append(
            RNNCell(input_size, hidden_size, bias, nonlinearity, device, dtype)
        )
        for i in range(1, self.num_layers):
            self.rnn_cells.append(
                RNNCell(hidden_size, hidden_size, bias, nonlinearity, device, dtype)
            )

        ### END YOUR SOLUTION

    def forward(self, X, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        seq_len = X.shape[0]
        if h0 is None:
            h0 = init.zeros(
                self.num_layers,
                X.shape[1],
                self.hidden_size,
                device=X.device,
                dtype=X.dtype,
            )
        h_input = list(ops.split(h0, 0))
        X_input = list(ops.split(X, 0))

        for i in range(seq_len):
            for j in range(self.num_layers):
                X_input[i] = self.rnn_cells[j](X_input[i], h_input[j])
                h_input[j] = X_input[i]
        output = ops.stack(X_input, 0)
        h_n = ops.stack(h_input, 0)

        return output, h_n
        ### END YOUR SOLUTION


class LSTMCell(Module):
    def __init__(
        self, input_size, hidden_size, bias=True, device=None, dtype="float32"
    ):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        bound = 1.0 / np.sqrt(hidden_size)
        self.W_ih = Parameter(
            init.rand(
                input_size,
                4 * hidden_size,
                low=-bound,
                high=bound,
                dtype=dtype,
                device=device,
            )
        )
        self.W_hh = Parameter(
            init.rand(
                hidden_size,
                4 * hidden_size,
                low=-bound,
                high=bound,
                dtype=dtype,
                device=device,
            )
        )
        self.bias_ih = (
            Parameter(
                init.rand(
                    4 * hidden_size,
                    1,
                    low=-bound,
                    high=bound,
                    dtype=dtype,
                    device=device,
                )
            )
            if bias
            else None
        )
        self.bias_hh = (
            Parameter(
                init.rand(
                    4 * hidden_size,
                    1,
                    low=-bound,
                    high=bound,
                    dtype=dtype,
                    device=device,
                )
            )
            if bias
            else None
        )
        self.sigmoid = Sigmoid()
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        if h is None:
            h0 = init.zeros(
                X.shape[0], self.hidden_size, dtype=X.dtype, device=X.device
            )
            c0 = init.zeros(
                X.shape[0], self.hidden_size, dtype=X.dtype, device=X.device
            )
        else:
            h0, c0 = h
        Z = X @ self.W_ih + h0 @ self.W_hh
        if self.bias:
            bias = self.bias_ih + self.bias_hh
            bias = ops.broadcast_to(ops.reshape(bias, (1, -1)), Z.shape)
            Z += bias
        Z = Z.reshape((X.shape[0], 4, self.hidden_size))
        i, f, g, o = ops.split(Z, 1)
        i, f, g, o = self.sigmoid(i), self.sigmoid(f), ops.tanh(g), self.sigmoid(o)
        c = f * c0 + i * g
        h = o * ops.tanh(c)
        return h, c
        ### END YOUR SOLUTION


class LSTM(Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
        bias=True,
        device=None,
        dtype="float32",
    ):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        ### BEGIN YOUR SOLUTION
        self.lstm_cells = []
        self.lstm_cells.append(LSTMCell(input_size, hidden_size, bias, device, dtype))
        for i in range(1, num_layers):
            self.lstm_cells.append(
                LSTMCell(hidden_size, hidden_size, bias, device, dtype)
            )

        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        seq_len, bs, _ = X.shape
        num_layers = len(self.lstm_cells)
        hidden_size = self.lstm_cells[0].W_hh.shape[0]
        if h is None:
            h0 = init.zeros(num_layers, bs, hidden_size, device=X.device, dtype=X.dtype)
            c0 = init.zeros(num_layers, bs, hidden_size, device=X.device, dtype=X.dtype)
        else:
            h0, c0 = h
        h_input = list(ops.split(h0, 0))
        c_input = list(ops.split(c0, 0))
        X_input = list(ops.split(X, 0))
        for i in range(seq_len):
            for j in range(num_layers):
                X_input[i], c_input[j] = self.lstm_cells[j](
                    X_input[i], (h_input[j], c_input[j])
                )
                h_input[j] = X_input[i]
        output = ops.stack(X_input, 0)
        h_n = ops.stack(h_input, 0)
        c_n = ops.stack(c_input, 0)
        return output, (h_n, c_n)
        ### END YOUR SOLUTION


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(
            init.randn(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )

        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        ### BEGIN YOUR SOLUTION
        one_hot = init.one_hot(self.weight.shape[0], x, device=x.device, dtype=x.dtype)
        seq_len, bs, num_embeddings = one_hot.shape
        one_hot = one_hot.reshape((seq_len * bs, num_embeddings))

        return ops.matmul(one_hot, self.weight).reshape(
            (seq_len, bs, self.weight.shape[1])
        )

        ### END YOUR SOLUTION
