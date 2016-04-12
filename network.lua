require 'models/GloVeEmbedding'
--nngraph.setDebug('true')

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
	return transfer_data(module)
end

function window_network_probs()
	local x                = nn.Identity()()
	
	local word_embeddings

	if params.use_embeddings then
		word_embeddings = GloVeEmbeddingFixed(data.vocab_map, 300, '')
	else
		word_embeddings  = nn.LookupTable(params.vocab_size, params.embedding_size)
	end

	local words = word_embeddings(x)

	local words_cat = nn.Reshape(params.layer_size[1])(words)

	local a = words_cat
	for i = 1, params.layers-1 do
		local z = nn.Linear(params.layer_size[i], params.layer_size[i+1])(a)
		a = nn.Tanh()(z)
	end

	local module           = nn.gModule({x},{a})
	module:getParameters():uniform(-params.init_weight, params.init_weight)
	return transfer_data(module)
end

function sll()
	local net_out	= nn.Identity()()
	local prev_d	= nn.Identity()()
	local A			= nn.Identity()()

	local net_out_t = nn.Transpose({1,2})(net_out)
	local rep_sum = nn.CAddTable()({net_out_t, prev_d})
	local sum_shp = nn.Replicate(params.num_tags,2)(rep_sum)
	local sum_mat = nn.CAddTable()({sum_shp,A})
	local lse_mean = nn.Mean(1)(sum_mat)
	local lse_mean_rep = nn.Replicate(params.num_tags,1)(lse_mean)
	local lse_mean_norm = nn.CSubTable()({sum_mat, lse_mean_rep})
	local next_d = nn.Log()(nn.Sum(1)(nn.Exp()(lse_mean_norm)))
	local next_d = nn.CAddTable()({next_d, lse_mean})

	local module = nn.gModule({net_out, prev_d, A}, {next_d})
	return transfer_data(module)
end


function sll_network()
	local x		= nn.Identity()()
	local prev	= nn.Identity()()
	local A		= nn.Identity()()
	
	local word_embeddings

	if params.use_embeddings then
		word_embeddings = GloVeEmbeddingFixed(data.vocab_map, 300, '')
	else
		word_embeddings  = nn.LookupTable(params.vocab_size, params.embedding_size)
	end

	local words = word_embeddings(x)

	local words_cat = nn.Reshape(params.batch_size,params.layer_size[1])(words)

	local a = words_cat
	for i = 1, params.layers-1 do
		local z = nn.Linear(params.layer_size[i], params.layer_size[i+1])(a)
		a = nn.Tanh()(z)
	end

	local sum_mat = nn.CAddTable()({nn.Reshape(params.num_tags,2)(nn.CAddTable()({a,prev})),A})
	local b = nn.Log()(nn.Sum(1)(nn.Exp()(sum_mat)))

	--local pred = nn.LogSoftMax()(a)
	
	local module           = nn.gModule({x, prev, A},{a, b})
	module:getParameters():uniform(-params.init_weight, params.init_weight)
	return transfer_data(module)
end

