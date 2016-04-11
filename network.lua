import 'models/GloVeEmbedding'

function window_network()
	local x                = nn.Identity()()
	
	local word_embeddings

	if params.use_embeddings then
		word_embeddings = GloVeEmbeddingFixed(data.vocab_map, 300, '')
	else
		word_embeddings  = nn.LookupTable(params.vocab_size,params.embedding_size)
	end

	local words = word_embeddings(x)

	local words_cat = nn.Reshape(params.batch_size,params.layer_size[1])(words)

	local a = words_cat
	for i = 1, params.layers-1 do
		local z = nn.Linear(params.layer_size[i], params.layer_size[i+1])(a)
		a = nn.Tanh()(z)
	end

	local pred = nn.LogSoftMax()(a)
	
	local module           = nn.gModule({x},{pred})
	module:getParameters():uniform(-params.init_weight, params.init_weight)
	return module:cuda()
end

