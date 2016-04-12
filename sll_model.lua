
local sll_model = torch.class('sll_model')
function sll_model:__init()
	self.core_network = window_network_probs()
	self.sll_network  = sll()
	self.networks	= g_cloneManyTimes(self.core_network, params.seq_length)
	self.rnns		= g_cloneManyTimes(self.sll_network, params.seq_length)

	self.paramx, self.paramdx = self.core_network:getParameters()

	self.net_out	= {}
	self.sll		= {}
	for i = 1, params.seq_length do
		self.net_out[i] = transfer_data(torch.zeros(1, params.num_tags))
		self.sll[i]		= transfer_data(torch.zeros(1, params.num_tags))
	end
	self.ds		= transfer_data(torch.zeros(1, params.num_tags))
	self.A		= transfer_data(torch.randn(params.num_tags, params.num_tags))
	self.dA		= transfer_data(torch.randn(params.num_tags, params.num_tags))
end

function sll_model:fp(sequence, targets)
	local num_iter = sequence:size(2) - (params.window_size - 1)
	num_iter = math.min(num_iter, params.seq_length)
	local init = transfer_data(torch.zeros(params.num_tags))
	self.sll[0] = init
	local prob = 0
	for i = 1,num_iter do
		local x = sequence:sub(1,1,i,i+params.window_size-1)
		local y = targets[i]
		self.net_out[i] = self.networks[i]:forward(x)
		if i > 1 then
			self.sll[i]	= self.rnns[i]:forward({self.net_out[i], self.sll[i-1], self.A})
		else
			self.sll[i] = self.net_out[1]
		end
		--graph.dot(self.rnns[1].fg, 'Forward Graph','./fg')
		--print(self.net_out[i]:size())
		prob = prob + self.net_out[i][1][y]
		if i > 1 then prob = prob + self.A[targets[i-1]][targets[i]] end
	end
	--print(self.sll)
	--print('ni',num_iter)
	--print(#self.sll)
	local err = -(prob - self.sll[num_iter]:sum())
	return err
end

function sll_model:bp(sequence, targets)
	local num_iter = sequence:size(2) - (params.window_size - 1)
	num_iter = math.min(num_iter, params.seq_length)
	self.paramdx:zero()
	self.dsll	= transfer_data(torch.ones(params.num_tags, 1))
	--self.dsll:fill(1)
	self.dA:fill(0)
	for i = num_iter,1,-1 do
		local tmp = self.rnns[i]:backward({self.net_out[i], self.sll[i-1], self.A}, self.dsll)

		local net_grad = tmp[1]
		self.dsll = tmp[2]
		self.dA:add(tmp[3])
		if i > 1 then self.dA[targets[i]][targets[i-1]] = self.dA[targets[i]][targets[i-1]] - 1 end

		local x = sequence:sub(1,1,i,i+params.window_size-1)
		local target = targets[i]
		--print(net_grad)
		net_grad[1][target] = net_grad[1][target] - 1

		self.networks[i]:backward({x}, net_grad)
	end
	self.A:add(self.dA:mul(-params.lr))
end

function sll_model:pass(sequence, targets)
	local perp = self:fp(sequence, targets)
	self:bp(sequence, targets)
	return perp
end

function sll_model:feval(x)

end
