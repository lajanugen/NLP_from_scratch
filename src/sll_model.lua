
local sll_model = torch.class('sll_model')
function sll_model:__init()
	self.core_network = window_network_probs()
	self.sll_network  = sll()

	self.paramx, self.paramdx = self.core_network:getParameters()

	if params.init_with_wll then
		local params = torch.load(params.init_with_wll_path)
		self.paramx:copy(params)
	elseif params.pre_init_mdl then
		local params = torch.load(params.pre_init_mdl_path .. params.pre_init_mdl_name)
		self.paramx:copy(params)
	end

	self.networks	= g_cloneManyTimes(self.core_network, params.seq_length)
	self.rnns		= g_cloneManyTimes(self.sll_network, params.seq_length)

	self.net_out	= {}
	self.sll		= {}
	for i = 1, params.seq_length do
		self.net_out[i] = transfer_data(torch.zeros(1, params.num_tags))
		self.sll[i]		= transfer_data(torch.zeros(1, params.num_tags))
	end
	self.ds		= transfer_data(torch.zeros(1, params.num_tags))
	self.A		= transfer_data(torch.zeros(params.num_tags, params.num_tags):uniform(-params.init_weight, params.init_weight))
	self.A0		= transfer_data(torch.zeros(params.num_tags)):uniform(-params.init_weight, params.init_weight)
	if params.A_grad then
		self.A:fill(0)
	end
	self.dA		= transfer_data(torch.zeros(params.num_tags, params.num_tags))
	self.dA0	= transfer_data(torch.zeros(params.num_tags))

	--self.LSE = transfer_data(nn.Sequential():add(nn.Exp()):add(nn.Sum()):add(nn.Log()))
	self.LSE = LSE()
	print('done_init')
end

function sll_model:fp(data, targets)
	sequence = data[1]
	sequence_caps = data[2]

	local num_iter = sequence:size(2) - (params.window_size - 1)
	num_iter = math.min(num_iter, params.seq_length)
	local init = transfer_data(torch.zeros(params.num_tags))
	self.sll[0] = init
	local prob = 0

	for i = 1,num_iter do
		local x = sequence:sub(1,1,i,i+params.window_size-1)
		local x_caps
		if params.cap_feat then x_caps = sequence_caps:sub(1,1,i,i+params.window_size-1) end

		local y = targets[i]
		if params.cap_feat then self.net_out[i] = self.networks[i]:forward({x, x_caps})
		else					self.net_out[i] = self.networks[i]:forward(x)
		end

		if i > 1 then	self.sll[i]	= self.rnns[i]:forward({self.net_out[i], self.sll[i-1], self.A})
		else			self.sll[i]:add(self.net_out[1], self.A0)
		end
		
		--graph.dot(self.rnns[1].fg, 'Forward Graph','./fg')
		prob = prob + self.net_out[i][1][y]
		if i > 1 then prob = prob + self.A[targets[i-1]][targets[i]] end
	end
	local err = -prob + self.LSE:forward(self.sll[num_iter]:t())[1] -- Negative log prob
	return err
end

function sll_model:bp(data, targets)
	sequence = data[1]
	sequence_caps = data[2]
	local num_iter = sequence:size(2) - (params.window_size - 1)
	num_iter = math.min(num_iter, params.seq_length)

	self.paramdx:zero()
	self.dsll = self.LSE:backward(self.sll[num_iter]:t(), transfer_data(torch.ones(1)))
	self.dA:fill(0)

	local net_grad
	for i = num_iter,1,-1 do

		local target = targets[i]

		if i > 1 then
			local tmp = self.rnns[i]:backward({self.net_out[i], self.sll[i-1], self.A}, self.dsll)
			net_grad = tmp[1]
			self.dsll = tmp[2]
			self.dA:add(tmp[3])
		else
			net_grad = self.dsll
		end

		if i > 1 then	self.dA[targets[i-1]][targets[i]] = self.dA[targets[i-1]][targets[i]] - 1
		else			
			self.dA0:copy(self.dsll)
			self.dA0[target] = self.dA0[target] - 1
		end

		local x = sequence:sub(1,1,i,i+params.window_size-1)
		local x_caps 
		if params.cap_feat then x_caps = sequence_caps:sub(1,1,i,i+params.window_size-1) end

		--print(net_grad:size(),target)
		net_grad[1][target] = net_grad[1][target] - 1

		if params.cap_feat then self.networks[i]:backward({x, x_caps}, net_grad)
		else					self.networks[i]:backward({x}, net_grad)
		end
	end
	if params.A_grad then
		--self.A:add(self.dA:mul(-params.lr))
		param_update(self.A, self.dA)
	end
	--self.paramx:add(self.paramdx:mul(-params.lr))
	--self.A0:add(self.dA0:mul(-params.lr))
	param_update(self.paramx, self.paramdx)
	param_update(self.A0, self.dA0)
end

function sll_model:pass(sequence, targets)
	local perp = self:fp(sequence, targets)
	self:bp(sequence, targets)
	return perp
end

function sll_model:run(tvt)
	local done = false
	local sequence, targets
	local num_cor, num_elem = 0, 0 while not done do 
	--for i = 1,100 do
		sequence, sequence_caps, targets, done = data:get_next_batch(tvt)

		local num_iter = sequence:size(2) - (params.window_size - 1)
		num_iter = math.min(num_iter, params.seq_length)

		local dp_inds = torch.Tensor(params.num_tags, num_iter)
		local dp_vals = transfer_data(torch.Tensor(params.num_tags, num_iter))
		local predicted_labels = torch.Tensor(num_iter)

		local init = transfer_data(torch.zeros(params.num_tags))
		self.sll[0] = init
		for i = 1,num_iter do
			local x = sequence:sub(1,1,i,i+params.window_size-1)
			local x_caps

			if params.cap_feat then x_caps = sequence_caps:sub(1,1,i,i+params.window_size-1) end
			--local y = targets[i]
			if params.cap_feat then self.net_out[i] = self.networks[i]:forward({x, x_caps})
			else					self.net_out[i] = self.networks[i]:forward(x)
			end

			if i > 1 then
				self.sll[i]	= self.sll_network:forward({self.net_out[i], self.sll[i-1], self.A})

				--require('mobdebug').on()
				local sum_mat = dp_vals[{{},{i-1}}]:repeatTensor(1,params.num_tags) + self.A
				--require('mobdebug').off()
				local max_vals, max_inds = torch.max(sum_mat, 1)
				dp_inds[{{},{i}}]:copy(max_inds)
				dp_vals[{{},{i}}]:copy(max_vals:cmul(self.net_out[i]))
			else
				self.sll[i]:add(self.net_out[1], self.A0)
				dp_vals[{{},{1}}]:copy(self.sll[1])
			end
		end
		local _, label = torch.max(dp_vals[{{},{num_iter}}], 1)
		label = label[1][1]
		predicted_labels[num_iter] = label
		for i = num_iter,2,-1 do
			label = dp_inds[label][i]
			predicted_labels[i-1] = label
		end
		if targets:size(1) > params.seq_length then targets = targets:sub(1,params.seq_length) end 
		-- Truncate sentence length to seq_length
		--print(predicted_labels:size(), targets:size(), data.sent_ptr[2])
		local num_correct = torch.eq(predicted_labels, targets:double()):sum()
		num_cor = num_cor + num_correct
		num_elem = num_elem + num_iter
	end
	print(num_elem, num_cor/num_elem)
	return num_cor/num_elem
end

function param_update(param, grad, state, coeff)
	if coeff then grad:copy(params.reg_coeff * param) end
	local coeff = 1 or coeff
	local norm_dw
	if params.backprop then
		norm_dw = grad:norm()
		if params.grad_clipping and (norm_dw > params.max_grad_norm) then
			local shrink_factor = params.max_grad_norm / norm_dw
			--if not params.custom_optimizer then --Vanilla GD
			grad:mul(shrink_factor) -- Gradient clipping
			--end
		end
		if params.update_params then
			if      params.custom_optimizer == 'adam'       then adam(param, grad, opt, state)
			elseif  params.custom_optimizer == 'rmsprop'    then rmsprop(param, grad, opt, state)
			else                                                 param:add(grad:mul(-params.lr))
			end
		end
	end
	return norm_dw
end

