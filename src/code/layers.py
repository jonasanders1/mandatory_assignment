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
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.w = np.random.uniform(-1, 1, (dim_in, dim_out)) / max(dim_in, dim_out)
        self.b = np.random.uniform(-1, 1, (dim_out,)) / max(dim_in, dim_out)
        self.dw = None
        self.db = None

    def forward(self, x):
        self.store = x
        out = np.dot(x, self.w) + self.b

        return out

    def backward(self, delta):
        self.dw = np.dot(self.store.T, delta) 
        self.db = np.sum(delta, axis=0) 
        # Compute the gradient for the input to propagate back
        dx = np.dot(delta, self.w.T)
        # Update the weights and bias using the computed gradients
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
        self.w = np.random.normal(0, 0.1, filtersize)
        self.b = np.random.normal(0, 0.1, (filtersize[0],))
        self.dw = None
        self.db = None

    def forward(self, x):
        """
        Forward pass of convolutional layer.
        
        x_col: Input tensor reshaped to matrix form.
        store_shape: Save shape of input tensor for backward pass.
        store_col: Save input tensor on matrix from for backward pass.
        """
        self.store = x
        N, C, H, W = x.shape
        
        # Calculate output dimensions
        Hout = int((H - self.filtersize[2] + 2*self.pad)/self.stride + 1)
        Wout = int((W - self.filtersize[3] + 2*self.pad)/self.stride + 1)
        
        # Reshape filters for matrix multiplication
        w_reshaped = self.w.reshape(self.filtersize[0], -1)
        
        # Convert input to columns using im2col
        x_cols = utils.im2col_indices(x, self.filtersize[2], self.filtersize[3], padding=self.pad, stride=self.stride)
        
        # Perform convolution as matrix multiplication
        out = np.dot(w_reshaped, x_cols) + self.b.reshape(-1, 1)
        
        # Reshape output
        out = out.reshape(self.filtersize[0], Hout, Wout, N)
        out = out.transpose(3, 0, 1, 2)
        
        # Store for backward pass
        self.x_cols = x_cols
        
        return out

    def backward(self, delta):
        """
        Backward pass of convolutional layer.
        
        delta: gradients from layer above
        dx: gradients that are propagated to layer below
        """
        x = self.store
        N, C, H, W = x.shape
        
        # Reshape delta
        delta_reshaped = delta.transpose(1, 2, 3, 0).reshape(self.filtersize[0], -1)
        
        # Compute gradient w.r.t weights
        self.dw = np.dot(delta_reshaped, self.x_cols.T)
        self.dw = self.dw.reshape(self.w.shape)
        
        # Compute gradient w.r.t bias
        self.db = np.sum(delta_reshaped, axis=1)
        
        # Compute gradient w.r.t input
        w_reshaped = self.w.reshape(self.filtersize[0], -1)
        dx_cols = np.dot(w_reshaped.T, delta_reshaped)
        
        # Convert back to image shape using col2im
        dx = utils.col2im_indices(dx_cols, x.shape, self.filtersize[2], self.filtersize[3], padding=self.pad, stride=self.stride)
        
        # Update weights and bias
        self.w -= update_param(self.dw)
        self.b -= update_param(self.db)
        
        return dx


class MaxPoolingLayer(Layers):
    """
    Implementation of MaxPoolingLayer.
    pool_r, pool_c: integers that denote pooling window size along row and column direction
    stride: integer that denotes with what stride the window is applied
    """
    def __init__(self, pool_height, pool_width, stride):
        self.pool_height = pool_height
        self.pool_width = pool_width
        self.stride = stride
        self.max_idx = None

    def forward(self, x):
        """
        Forward pass.
        x: Input tensor of form (NxCxHxW)
        out: Output tensor of form NxCxH_outxW_out
        N: Batch size
        C: Nr of channels
        H, H_out: Input and output heights
        W, W_out: Input and output width
        """
        N, C, H, W = x.shape
        
        out_height = (H - self.pool_height) // self.stride + 1
        out_width = (W - self.pool_width) // self.stride + 1
      
        out = np.zeros((N, C, out_height, out_width))
        self.max_idx = np.zeros((N, C, out_height, out_width, 2), dtype=np.int32)  # Store both indices
      
        for n in range(N):
            for c in range(C):
                for i in range(out_height):
                    for j in range(out_width):
                        h_start = i * self.stride
                        h_end = h_start + self.pool_height
                        w_start = j * self.stride
                        w_end = w_start + self.pool_width
                        
                        window = x[n, c, h_start:h_end, w_start:w_end]
                        max_val = np.max(window)
                        out[n, c, i, j] = max_val
                        
                        # Get local indices of max value
                        local_idx = np.unravel_index(window.argmax(), window.shape)
                        # Store global indices
                        self.max_idx[n, c, i, j] = [h_start + local_idx[0], w_start + local_idx[1]]
        
        self.store = x
        return out

    def backward(self, grad):
        """
        Backward pass.
        delta: loss derivative from above (of size NxCxH_outxW_out)
        dX: gradient of loss wrt. input (of size NxCxHxW)
        """
        N, C, H, W = grad.shape
        dx = np.zeros(self.store.shape)
        
        for n in range(N):
            for c in range(C):
                for i in range(H):
                    for j in range(W):
                        h_idx, w_idx = self.max_idx[n, c, i, j]
                        dx[n, c, h_idx, w_idx] = grad[n, c, i, j]

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
        next_h: Updated hidden state(Nxdim_hid)
        next_c: Updated cell state(Nxdim_hid)
        cache: A tuple where you can store anything that might be useful for the backward pass
        """
        
	    ######################################################
        ######## REPLACE NEXT PART WITH YOUR SOLUTION ########
        ######################################################
        next_h = np.random.random_sample(h.shape)
        next_c = np.random.random_sample(c.shape)
        cache = (next_h, next_c)
        ######################################################
        ######################################################
        ######################################################

        return next_h, next_c, cache

    def backward_step(self, delta_h, delta_c, store):
        """
        Implementation of a single backward step (one timestep)
        delta_h: Upstream gradients from hidden state
        delta_h: Upstream gradients from cell state
        store:
          hn: Updated hidden state from forward pass (Nxdim_hid) where dim_hid=#hidden units
          x: Input to layer (Nxdim_in) where N=#samples in batch and dim_in=feature dimension
          h: Hidden state from previous time step (Nxdim_hid) where dim_hid=#hidden units
          cn: Updated cell state from forward pass (Nxdim_hid) where dim_hid=#hidden units
          c: Cell state from previous time step (Nxdim_hid) where dim_hid=#hidden units
          cache: Whatever was added to the cache in forward pass
        dx: Gradient of loss wrt. input
        dh: Gradient of loss wrt. previous hidden state
        dc: Gradient of loss wrt. previous cell state
        dwh: Gradient of loss wrt. weight tensor for hidden to hidden mapping
        dwx: Gradient of loss wrt. weight tensor for input to hidden mapping
        db: Gradient of loss wrt. bias vector
        """
        hn, x, h, cn, c, cache = store

        ######################################################
        ######## REPLACE NEXT PART WITH YOUR SOLUTION ########
        ######################################################
        dx = np.random.random_sample(x.shape)
        dh = np.random.random_sample(h.shape)
        dc = np.random.random_sample(c.shape)
        dwh = np.random.random_sample(self.wh.shape)
        dwx = np.random.random_sample(self.wx.shape)
        db = np.random.random_sample(self.b.shape)
        ######################################################
        ######################################################
        ######################################################

        return dx, dh, dc, dwh, dwx, db


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
        x: Input to layer. Any dimension.
        """
                
        self.store = x
        out = np.maximum(0, x)

        return out

    def backward(self, delta):
        """
        delta: Loss derivative from above. Any dimension.
        """
        dx = delta * (self.store > 0)
        return dx
