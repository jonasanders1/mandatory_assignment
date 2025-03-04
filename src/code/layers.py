import numpy as np
import utils


def update_param(dx, learning_rate=1e-2):
    """
    Implementation of standard gradient descent algorithm.
    """
    return learning_rate * dx

def update_param_adagrad(dx, mx, learning_rate=1e-2):
    """
    Implementation of adagrad algorithm.
    """
    return learning_rate * dx / np.sqrt(mx+1e-8)

def sigmoid(x):
    """
    A numerically stable version of the logistic sigmoid function.
    """
    pos_mask = (x >= 0)
    neg_mask = (x < 0)
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)

class Layers():
    def __init__(self):
        """
        store: used to store variables and pass information from forward to backward pass. 
        """
        self.store = None

class FullyConnectedLayer(Layers):
    def __init__(self, dim_in, dim_out):
        """
        Implementation of a fully connected layer.

        dim_in: Number of neurons in previous layer.
        dim_out: Number of neurons in current layer.
        w: Weight matrix of the layer.
        b: Bias vector of the layer.
        dw: Gradient of weight matrix.
        db: Gradient of bias vector
        """
        self.dim_in = dim_in
        self.dim_out = dim_out
        
    
        self.w = np.random.randn(dim_in, dim_out) * np.sqrt(2.0 / (dim_in + dim_out))
        # self.w = np.random.uniform(-1, 1, (dim_in, dim_out)) / max(dim_in, dim_out) # * OLD
        
        self.b = np.random.uniform(-1, 1, (dim_out,)) / max(dim_in, dim_out)
        
        self.dw = None
        self.db = None

    def forward(self, x):
        """
        Forward pass of fully connected layer.
        x: Input to layer (either of form Nxdim_in or in tensor form after convolution NxCxHxW).
        """
        # Reshape input if it comes from conv layer
        if len(x.shape) > 2:
            N = x.shape[0]
            x = x.reshape(N, -1)
        
        self.store = x
        out = np.dot(x, self.w) + self.b
        return out

    def backward(self, delta):
        """
        Backward pass of fully connected layer.
        delta: Error from succeeding layer
        """
        # Compute gradients
        self.dw = np.dot(self.store.T, delta)
        self.db = np.sum(delta, axis=0)
        dx = np.dot(delta, self.w.T)
        
        # Update parameters
        self.w -= update_param(self.dw)
        self.b -= update_param(self.db)
        
        return dx


class ConvolutionalLayer(Layers):
    def __init__(self, filtersize, pad=0, stride=1):
        """
        Implementation of a convolutional layer.

        filtersize = (C_out, C_in, F_H, F_W)
        w: Weight tensor of layer.
        b: Bias vector of layer.
        dw: Gradient of weight tensor.
        db: Gradient of bias vector
        """
        self.filtersize = filtersize
        self.pad = pad
        self.stride = stride
        
        
        # self.w = np.random.normal(0, 0.1, filtersize) # * OLD
        self.w = np.random.randn(*filtersize) * np.sqrt(2.0 / (filtersize[1] * filtersize[2] * filtersize[3]))
        self.b = np.random.normal(0, 0.1, (filtersize[0],))
        
        
        
        self.dw = None
        self.db = None

    def forward(self, x):
        """
        Forward pass of convolutional layer.
        x: Input tensor of form (NxCxHxW)
        """
        N, C, H, W = x.shape
        F, C, HH, WW = self.filtersize
        
        # Calculate output dimensions
        Hout = int((H - HH + 2*self.pad)/self.stride + 1)
        Wout = int((W - WW + 2*self.pad)/self.stride + 1)
        
        # im2col transformation
        x_cols = utils.im2col_indices(x, HH, WW, padding=self.pad, stride=self.stride)
        w_cols = self.w.reshape(F, -1)
        
        # Compute convolution as matrix multiplication
        out = np.dot(w_cols, x_cols) + self.b.reshape(-1, 1)
        out = out.reshape(F, Hout, Wout, N)
        out = out.transpose(3, 0, 1, 2)
        
        self.store = x
        self.x_cols = x_cols
        return out

    def backward(self, delta):
        """
        Backward pass of convolutional layer.
        delta: gradients from layer above
        """
        x = self.store
        N, C, H, W = x.shape
        F, C, HH, WW = self.filtersize
        
        # Reshape delta
        delta = delta.transpose(1, 2, 3, 0).reshape(F, -1)
        
        # Compute gradients
        self.dw = np.dot(delta, self.x_cols.T).reshape(self.w.shape)
        self.db = np.sum(delta, axis=1)
        
        # Compute gradient w.r.t. input
        dx_cols = np.dot(self.w.reshape(F, -1).T, delta)
        dx = utils.col2im_indices(dx_cols, x.shape, HH, WW, padding=self.pad, stride=self.stride)
        
        return dx


class MaxPoolingLayer(Layers):
    def __init__(self, pool_r, pool_c, stride):
        self.pool_r = pool_r
        self.pool_c = pool_c
        self.stride = stride

    def forward(self, x):
        N, C, H, W = x.shape
        pool_height = self.pool_r
        pool_width = self.pool_c
        stride = self.stride
        
        H_out = H // pool_height
        W_out = W // pool_width
        
        # Reshape input for pooling
        x_reshaped = x.reshape(N, C, H_out, pool_height, W_out, pool_width)
        
        # Perform max pooling
        out = x_reshaped.max(axis=3).max(axis=4)
        
        self.store = x
        self.x_reshaped = x_reshaped
        return out

    def backward(self, delta):
        x = self.store
        x_reshaped = self.x_reshaped
        N, C, H_out, W_out = delta.shape
        
        # Reshape delta to match the pooling dimensions
        delta_reshaped = delta.reshape(N, C, H_out, 1, W_out, 1)
        
        # Create mask for maximum values
        max_mask = (x_reshaped == x_reshaped.max(axis=3, keepdims=True).max(axis=5, keepdims=True))
        
        # Distribute gradient
        dx_reshaped = max_mask * delta_reshaped
        dx = dx_reshaped.reshape(self.store.shape)
        
        return dx


class LSTMLayer(Layers):
    """
    Implementation of a LSTM layer.

    dim_in: Integer indicating input dimension
    dim_hid: Integer indicating hidden dimension
    wx: Weight tensor for input to hidden mapping (dim_in, 4*dim_hid)
    wh: Weight tensor for hidden to hidden mapping (dim_hid, 4*dim_hid)
    b: Bias vector of layer (4*dim_hid)
    """
    def __init__(self, dim_in, dim_hid):
        self.dim_in = dim_in
        self.dim_hid = dim_hid
        self.wx = np.random.normal(0, 0.1, (dim_in, 4*dim_hid))
        self.wh = np.random.normal(0, 0.1, (dim_hid, 4*dim_hid))
        self.b = np.random.normal(0, 0.1, (4*dim_hid,))

    def forward_step(self, x, h, c):
        """
        Implementation of a single forward step (one timestep)
        x: Input to layer (Nxdim_in) where N=#samples in batch and dim_in=feature dimension
        h: Hidden state from previous time step (Nxdim_hid) where dim_hid=#hidden units
        c: Cell state from previous time step (Nxdim_hid) where dim_hid=#hidden units
        
        Returns:
        next_h: Updated hidden state (Nxdim_hid)
        next_c: Updated cell state (Nxdim_hid)
        cache: Tuple containing values needed for backward pass
        """
        # Compute all gate inputs (dot products and bias)
        gates = np.dot(x, self.wx) + np.dot(h, self.wh) + self.b
        
        # Split gates into their components
        n = self.dim_hid
        i = sigmoid(gates[:, :n])          # input gate
        f = sigmoid(gates[:, n:2*n])       # forget gate
        o = sigmoid(gates[:, 2*n:3*n])     # output gate
        g = np.tanh(gates[:, 3*n:])        # cell gate
        
        # Update cell state
        next_c = f * c + i * g
        
        # Update hidden state
        next_h = o * np.tanh(next_c)
        
        # Cache values needed for backward pass
        cache = (i, f, o, g, x, h, c)
        
        return next_h, next_c, cache

    def backward_step(self, delta_h, delta_c, store):
        """
        Implementation of a single backward step (one timestep)
        delta_h: Upstream gradients from hidden state
        delta_c: Upstream gradients from cell state
        store: Tuple containing cached values from forward pass
            (i, f, o, g, next_c, next_h, x, h, c)
        
        Returns:
        dx: Gradient of loss wrt input
        dh: Gradient of loss wrt previous hidden state
        dc: Gradient of loss wrt previous cell state
        dwh: Gradient of loss wrt weight tensor for hidden to hidden mapping
        dwx: Gradient of loss wrt weight tensor for input to hidden mapping
        db: Gradient of loss wrt bias vector
        """
        # Unpack stored variables from forward pass
        next_h, x, h, next_c, c, (i, f, o, g, x, h, prev_c) = store
        
        # Gradient of hidden state
        do = delta_h * np.tanh(next_c)
        dnext_c = delta_c + delta_h * o * (1 - np.tanh(next_c)**2)
        
        # Gradient of cell state
        di = dnext_c * g
        df = dnext_c * prev_c
        dg = dnext_c * i
        dprev_c = dnext_c * f
        
        # Gradient of gate inputs
        di_input = di * i * (1 - i)
        df_input = df * f * (1 - f)
        do_input = do * o * (1 - o)
        dg_input = dg * (1 - g**2)
        
        # Concatenate gate gradients
        dgates = np.concatenate((di_input, df_input, do_input, dg_input), axis=1)
        
        # Compute gradients of weights and biases
        dwx = np.dot(x.T, dgates)
        dwh = np.dot(h.T, dgates)
        db = np.sum(dgates, axis=0)
        
        # Compute gradients for recurrent inputs
        dx = np.dot(dgates, self.wx.T)
        dh = np.dot(dgates, self.wh.T)
        
        return dx, dh, dprev_c, dwh, dwx, db


class WordEmbeddingLayer(Layers):
    """
    Implementation of WordEmbeddingLayer.
    """
    def __init__(self, vocab_dim, embedding_dim):
        self.w = np.random.normal(0, 0.1, (vocab_dim, embedding_dim))
        self.dw = None

    def forward(self, x):
        """
        Forward pass.
        Look-up embedding for x of form (NxTx1) where each element is an integer indicating the word id.
        N: Number of words in batch. 
        T: Number of timesteps.
        Output: (NxTxE) where E is embedding dimensionality.
        """
        self.store = x
        return self.w[x,:]

    def backward(self, delta):
        """
        Backward pass. Update embedding matrix.
        Delta: Loss derivative from above
        """
        x = self.store
        self.dw = np.zeros(self.w.shape)
        np.add.at(self.dw, x, delta)
        self.w -= update_param(self.dw)
        return 0


"""
Activation functions
"""
class SoftmaxLossLayer(Layers):
    """
    Implementation of SoftmaxLayer forward pass with cross-entropy loss.
    """
    def forward(self, x, y):
        ex = np.exp(x-np.max(x, axis=1, keepdims=True))
        y_hat = ex/np.sum(ex, axis=1, keepdims=True)
        m = y.shape[0]
        log_likehood = -np.log(y_hat[range(m), y.astype(int)])
        loss = np.sum(log_likehood) / m

        d_out = y_hat
        d_out[range(m), y.astype(int)] -= 1
        d_out /= m

        return loss, d_out

class SoftmaxLayer(Layers):
    """
    Implementation of SoftmaxLayer forward pass.
    """
    def forward(self, x):
        ex = np.exp(x-np.max(x, axis=1, keepdims=True))
        y_hat = ex/np.sum(ex, axis=1, keepdims=True)
        return y_hat


class ReluLayer(Layers):
    """
    Implementation of relu activation function.
    """
    def forward(self, x):
        """
        Forward pass of ReLU activation.
        x: Input to layer
        """
        self.store = x
        return np.maximum(0, x)

    def backward(self, delta):
        """
        Backward pass of ReLU activation.
        delta: Loss derivative from above
        """
        # Ensure delta and store have same shape
        if delta.shape != self.store.shape:
            delta = delta.reshape(self.store.shape)
        
        dx = delta * (self.store > 0)
        return dx
