
function window_networks()
	local x                = nn.Identity()()
	
	local word_embeddings  = nn.LookupTable(params.vocab_size,parans.embedding_size)

	local words = word_embeddings(x)

	local words_cat = nn.Reshape(params.batch_size,1,params.window_size*params.embedding_size)

	local z,a
	local input_size, output_size = params.window_size*params.embedding_size
	for i = 1, params.layers do
		output_size = params.layer_size[i]
		z = nn.Linear(input_size, params.hid_size)(words_cat)
		a = nn.Tanh()(z1)
		input_size = output_size
	end

	local pred = nn.LogSoftMax()(a)
	local err  = nn.ClassNLLCriterion()({pred, y})
	
	local module           = nn.gModule({x,y},{err})
	module:getParameters():uniform(-params.init_weight, params.init_weight)
	return transfer_data(module)
end

