require 'data'
require 'network'
require 'nn'
require 'nngraph'

params = {
window_size		= 5,
vocab_size		= 100,
batch_size		= 1,
layers			= 1,
num_tags		= 1,
lr				= 1,
layers			= 3,
layer_size		= {5, 30, 5},
objective		= 'sll',
custom_optimizer= false,
init_weight		= 0.01,
max_max_epoch	= 100,
stats_freq		= 1,
log_err			= true,
log_err_freq	= 1,
err_log			= {false,true,true},
embedding_size  = 10,
use_embeddings  = false,
caps_feats		= true,
senna_vocab		= true,
seq_length		= 100,
use_gpu			= false,
dummy_data		= false
}

params.layer_size[1] = params.window_size * params.embedding_size
function transfer_data(x)
	if params.use_gpu then	return x:cuda()
	else					return x
	end
end

data = data()

for i = 1,5000 do
	a, b = data:get_next_batch(1)
	--if i == 4810 then print(a) end
end
