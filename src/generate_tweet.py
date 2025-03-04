import sys
sys.path.append('code/')  # Add the code directory to Python path

import numpy as np
from rnn import RNN
from HE_tweets_loader import vecToWords

def generate_tweet():
    # Initialize model with same parameters as training
    model = RNN(embedding_dim=256, dim_hid=256, vocab_size=45)
    
    # Load the trained weights
    model.load_model('trump_model.npz')
    
    # Initialize starting states
    batch_size = 1
    h = np.zeros((batch_size, model.dim_hid))
    c = np.zeros((batch_size, model.dim_hid))
    
    # Create random starting token
    x = np.array([[np.random.randint(0, model.vocab_size)]])
    
    # Generate tweet
    print("\nGenerated Tweet:")
    model.test(x, h, c, dataset='trump')

if __name__ == "__main__":
    generate_tweet() 