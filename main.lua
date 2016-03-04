require 'nngraph'
require 'cunn'
require 'networks'

params = {
window_size		= 5,
vocab_size		= 100,
embedding_size  = 100,
batch_size		= 100,
layers			= 1,
num_tags		= 1,
layer_size		= {300, num_tags},
}


