require 'models/GloVeEmbedding'
--nngraph.setDebug('true')

function window_network()
	local x			= nn.Identity()()
	local x_caps	= nn.Identity()()
	
	local word_embeddings

	if params.use_embeddings then
		if params.embeddings_fixed then
			word_embeddings = GloVeEmbeddingFixed(data.vocab_map, 300, '')
		else
			word_embeddings = GloVeEmbedding(data.vocab_map, 300, '')
		end
	else
		word_embeddings  = nn.LookupTable(params.vocab_size,params.embedding_size)
	end

	local words_caps_cat
	local words_caps
	if params.cap_feat then
		words_caps = nn.LookupTable(5,params.caps_embedding_size)(x_caps)
		words_caps_cat = nn.Reshape(params.batch_size,params.window_size*params.caps_embedding_size)(words_caps)
	end

	local words = word_embeddings(x)
	local words_cat = nn.Reshape(params.batch_size,params.window_size*params.embedding_size)(words)

	local allfeatures
	if params.cap_feat then	allfeatures = nn.JoinTable(2,2)({words_cat, words_caps_cat})
	else					allfeatures = words_cat
	end

	local a = allfeatures
	for i = 1, params.layers-1 do
		local z = nn.Linear(params.layer_size[i], params.layer_size[i+1])(a)
		--a = nn.HardTanh()(z)
		a = nn.Tanh()(z)
	end

	local pred = nn.LogSoftMax()(a)
	
	local module
	if params.cap_feat then	module = nn.gModule({x, x_caps},{pred})
	else					module = nn.gModule({x},{pred})
	end
	if not params.use_embeddings then
		module:getParameters():uniform(-params.init_weight, params.init_weight)
	else
		p,g = module:parameters()
		for i = 1,#p do
			if p[i]:size(1) ~= 130003 then
				p[i]:uniform(-params.init_weight, params.init_weight)
			else
				print(p[i]:size())
			end
		end
	end
	return transfer_data(module)
end

function window_network_probs()
	local x                = nn.Identity()()
	local x_caps           = nn.Identity()() 

	local word_embeddings

	if params.use_embeddings then
		if params.embeddings_fixed then
			word_embeddings = GloVeEmbeddingFixed(data.vocab_map, 300, '')
		else
			word_embeddings = GloVeEmbedding(data.vocab_map, 300, '')
		end
	else
		word_embeddings  = nn.LookupTable(params.vocab_size,params.embedding_size)
	end

	local words_caps_cat
	local words_caps
	if params.cap_feat then
		words_caps = nn.LookupTable(5,params.caps_embedding_size)(x_caps)
		words_caps_cat = nn.Reshape(params.window_size*params.caps_embedding_size)(words_caps)
	end

	local words = word_embeddings(x)
	local words_cat = nn.Reshape(params.window_size*params.embedding_size)(words)

	local allfeatures
	if params.cap_feat then	allfeatures = nn.JoinTable(2,2)({words_cat, words_caps_cat})
	else					allfeatures = words_cat
	end

	local a = allfeatures
	for i = 1, params.layers-1 do
		local z = nn.Linear(params.layer_size[i], params.layer_size[i+1])(a)
		--a = nn.HardTanh()(z)
		a = nn.Tanh()(z)
	end

	local out = a

	local module
	if params.cap_feat then	module = nn.gModule({x, x_caps},{out})
	else					module = nn.gModule({x},{out})
	end
	module:getParameters():uniform(-params.init_weight, params.init_weight)
	return transfer_data(module)
end

function sll()
	local net_out	= nn.Identity()()
	local prev_d	= nn.Identity()()
	local A			= nn.Identity()()

	local prev_d_t = nn.Transpose({1,2})(prev_d)
	--local rep_sum = nn.CAddTable()({net_out_t, prev_d})
	local sum_shp = nn.Replicate(params.num_tags,2)(prev_d_t)
	local sum_mat = nn.CAddTable()({sum_shp,A})
	local lse_mean = nn.Mean(1)(sum_mat)
	local lse_mean_rep = nn.Replicate(params.num_tags,1)(lse_mean)
	local lse_mean_norm = nn.CSubTable()({sum_mat, lse_mean_rep})
	local next_d = nn.Log()(nn.Sum(1)(nn.Exp()(lse_mean_norm)))
	local next_d = nn.CAddTable()({next_d, lse_mean})

	next_d = nn.CAddTable()({next_d, net_out})
	next_d = nn.Transpose({1,2})(next_d)

	local module = nn.gModule({net_out, prev_d, A}, {next_d})
	return transfer_data(module)
end

function LSE()
	local x = nn.Identity()()

	local mean = nn.Mean()(x)
	local mean_rep = nn.Replicate(params.num_tags,1)(mean)
	local mean_sub = nn.CSubTable()({x, mean_rep})
	local exp = nn.Exp()(mean_sub)
	local sum = nn.Sum()(exp)
	local log = nn.Log()(sum)
	local y = nn.CAddTable()({log, mean})

	local module = nn.gModule({x},{y})
	return transfer_data(module)
end

--function sll()
--	local net_out	= nn.Identity()()
--	local prev_d	= nn.Identity()()
--	local A			= nn.Identity()()
--
--	local net_out_t = nn.Transpose({1,2})(net_out)
--	local rep_sum = nn.CAddTable()({net_out_t, prev_d})
--	local sum_shp = nn.Replicate(params.num_tags,2)(rep_sum)
--	local sum_mat = nn.CAddTable()({sum_shp,A})
--	local lse_mean = nn.Mean(1)(sum_mat)
--	local lse_mean_rep = nn.Replicate(params.num_tags,1)(lse_mean)
--	local lse_mean_norm = nn.CSubTable()({sum_mat, lse_mean_rep})
--	local next_d = nn.Log()(nn.Sum(1)(nn.Exp()(lse_mean_norm)))
--	local next_d = nn.CAddTable()({next_d, lse_mean})
--
--	local module = nn.gModule({net_out, prev_d, A}, {next_d})
--	return transfer_data(module)
--end
