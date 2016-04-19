
local wll_model = torch.class('wll_model')
function wll_model:__init()
	self.core_network = window_network()
	self.criterion = transfer_data(nn.ClassNLLCriterion())
	self.paramx, self.paramdx = self.core_network:getParameters()

	--self.core_network:parameters()[1]:uniform(-0.5,0.5)
	--self.core_network:parameters()[2]:uniform(-0.5,0.5)
	--local fanin = params.layer_size[1]
	--self.core_network:parameters()[3]:uniform(-0.5*math.sqrt(1/fanin),0.5*math.sqrt(1/fanin))
	--self.core_network:parameters()[4]:uniform(-0.5*math.sqrt(1/fanin),0.5*math.sqrt(1/fanin))
	--local fanin = params.layer_size[2]
	--self.core_network:parameters()[5]:uniform(-0.5*math.sqrt(1/fanin),0.5*math.sqrt(1/fanin))
	--self.core_network:parameters()[6]:uniform(-0.5*math.sqrt(1/fanin),0.5*math.sqrt(1/fanin))

	if params.pre_init_mdl then
		local params = torch.load(params.pre_init_mdl_path .. params.pre_init_mdl_name)
		self.paramx:copy(params)
	end
end

function wll_model:run(tvt)
	local num_batches = data:get_batch_count(tvt)
	local err = 0
	local predictions = {}
	local words = {}
	for i = 1,num_batches do
		local data_x, data_x_caps, data_y = data:get_next_batch(tvt)
		local pred
		if params.cap_feat then	pred = self.core_network:forward({data_x, data_x_caps})
		else					pred = self.core_network:forward(data_x)
		end
		_, pred_y = torch.max(pred,2)
		err = err + torch.mean(torch.eq(data_y, pred_y))
		local word = data_x[{{},{(params.window_size + 1)/2}}]
		for j = 1,pred_y:size(1) do
			table.insert(predictions, pred_y[j][1])
			table.insert(words, data.ivocab_map[word[j][1]])
		end
	end
	--print(#predictions)
	print(err/num_batches)
	return err/num_batches, predictions, words
end

--function wll_model:feval(x)
--	if x ~= paramx then
--		paramx:copy(x)
--	end
--	local pred = self.core_network:forward(data_x)
--	local err = self.criterion:forward(pred, data_y)
--	local df_dw = self.criterion:backward(pred, data_y)
--	self.paramdx:fill(0)
--	self.core_network:backward(data_x,df_dw)
--
--	return err, self.paramdx
--end


