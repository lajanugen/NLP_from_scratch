require 'nngraph'
require 'cunn'
require 'network'
require 'data'
require 'optim'
require 'utils'

params = {
window_size		= 5,
vocab_size		= 100,
batch_size		= 500,
layers			= 1,
num_tags		= 1,
lr				= 0.01,
layers			= 3,
layer_size		= {0, 300},
objective		= 'wll',
custom_optimizer= true,
init_weight		= 0.01,
max_max_epoch	= 100,
stats_freq		= 1,
log_err			= true,
log_err_freq	= 1,
err_log			= {false,true,true},
embedding_size  = 300,
use_embeddings  = true
}
opt = {
optimizer		= 'adam',
learning_rate	= 0.001,
momentum		= 0.5
}

g_init_gpu(arg)
data = data()
params.layer_size[1] = params.window_size * params.embedding_size
table.insert(params.layer_size, params.target_size)

if params.objective == 'wll' then 
	model = window_network()
	criterion = nn.ClassNLLCriterion():cuda()
end

paramx, paramdx = model:getParameters()

local function feval(x)
	if x ~= paramx then
		paramx:copy(x)
	end
	--assert(torch.sum(torch.eq(data_x,0)) == 0)
	--assert(torch.sum(torch.eq(data_y,0)) == 0) 
	--print(data_x)
	--print(params.vocab_size)
	local pred = model:forward(data_x)
	local err = criterion:forward(pred, data_y)
	local df_dw = criterion:backward(pred, data_y)
	paramdx:fill(0)
	model:backward(data_x,df_dw)

	return err, paramdx
end

local function run(tvt)
	local num_batches = data:get_batch_count(tvt)
	local err = 0
	for i = 1,num_batches do
		local data_x, data_y = data:get_next_batch(tvt)
		local pred = model:forward(data_x)
		_, pred_y = torch.max(pred,2)
		err = err + torch.mean(torch.eq(data_y, pred_y))
	end
	print(err/num_batches)
	return err/num_batches
end

local function run_flow() 
	local optim_config
	if		opt.optimizer=='rmsprop' then 	optim_config = {learningRate = opt.learning_rate, alpha = opt.decay_rate}
	elseif	opt.optimizer=='adagrad' then	optim_config = {learningRate = opt.learning_rate}
	elseif	opt.optimizer=='sgd'	 then 	optim_config = {learningRate = opt.learning_rate, momentum = opt.momentum}
	elseif	opt.optimizer=='adam'	 then 	optim_config = {learningRate = opt.learning_rate}
	else 	error('undefined optimizer')
	end

	print("Network parameters:")
	--print(params)

	--local step = 0
	step = 0
	local epoch = 0
	local total_cases = 0
	local beginning_time = torch.tic()
	local start_time = torch.tic()
	print("Starting training.")

	local epoch_size = data:get_batch_count(1)
	print('epoch_size',epoch_size)

	local errors = {{},{},{}}
	local train_errors = {}
	local perps, n_elems
	local perp, n_elem

	while epoch < params.max_max_epoch do
		--print(step)

		data_x, data_y = data:get_next_batch(1,false)

		if params.custom_optimizer then
			if		opt.optimizer=='rmsprop' then	_, loss = optim.rmsprop(feval, paramx, optim_config)
			elseif	opt.optimizer=='adagrad' then	_, loss = optim.adagrad(feval, paramx, optim_config)
			elseif	opt.optimizer=='sgd'	 then	_, loss = optim.sgd(feval, paramx, optim_config)
			elseif	opt.optimizer=='adam'	 then 	_, loss = optim.adam(feval, paramx, optim_config)
			else	error('undefined optimizer')
			end
			perp = loss[1]
			--n_elem = params.batch_size
		else
			perp,n_elem = model:pass()
		end

		if perps == nil then
			perps   = torch.zeros(epoch_size):add(perp)
			--n_elems = torch.zeros(epoch_size):add(n_elem)
		end
		perps[step % epoch_size + 1] = perp
		--n_elems[step % epoch_size + 1] = n_elem

		step = step + 1
		total_cases = total_cases + params.batch_size
		epoch = step / epoch_size
		if step % torch.round(epoch_size * params.stats_freq) == 0 then
			if data.timing then
				print(model.get_batch)
				print(model.enc_fp_time)
				print(model.dec_fp_time)
				print(model.enc_bp_time)
				print(model.dec_bp_time)
				model:reset_timers()
			end
			local wps = torch.floor(total_cases / torch.toc(start_time))
			local since_beginning = g_f3(torch.toc(beginning_time) / 60)
			--', train perp. = ' .. g_f3(torch.exp(perps:mean())) ..
			--print('word_embs',torch.sum(word_embs))
			local perp = perps:mean()
			print('epoch = ' .. g_f3(epoch) ..
			', train error = ' .. g_f3(perp) ..
			', wps = ' .. wps ..
			--', dw:norm() = ' .. g_f3(model.norm_dw) ..
			', lr = ' ..  g_f3(params.lr) ..
			', since beginning = ' .. since_beginning .. ' mins.')
		end
		if step % epoch_size == 0 then
			table.insert(train_errors,perps:mean())
			if params.lr_decay and (epoch > params.max_epoch) then
				params.lr = params.lr / params.decay
			end
		end
		if params.log_err and (step > 0) and (step % (epoch_size*params.log_err_freq) == 0) then
			if params.err_display then
				for t = 1,perps:size(1) do io.write(g_d(perps[t])) io.write(' ') end
				io.write('\n')
			end
			for i = 1,3 do if params.err_log[i] then table.insert(errors[i],run(i)) end end
		end

		if step % 33 == 0 then
			cutorch.synchronize()
			--collectgarbage()
		end
		if params.model_checkpoint and (step > 0) and ((step % (params.model_checkpoint_freq*epoch_size)) == 0) then
			if params.bwd then
				torch.save(params.save_path .. 'models/bwd_model' .. tostring(epoch),model);
			else
				torch.save(params.save_path .. 'models/paramx' .. tostring(epoch),model.paramx);
				torch.save(params.save_path .. 'models/model' .. tostring(epoch),model);
			end
		end
	end

	--ProFi:stop()
	--ProFi:writeReport('profiling_report')

	if params.save_models then
		if params.bwd then
			torch.save(params.save_path .. 'models/bwd_model',model);
		else
			torch.save(params.save_path .. 'models/model.paramx',model.paramx);
			torch.save(params.save_path .. 'models/model',model.paramx);
		end
	end

	if params.make_plots then
		plots(train_errors,'Training Error')
		for i = 1,3 do
			if params.err_log[i]  then 
				plots(errors[i],TRAIN_VAL_TEST[i] .. ' Error') 
				log_to_file(errors[i],params.save_path .. '/logs/err/' .. TRAIN_VAL_TEST[i])
			end
			if params.log_bleu and params.bleu_log[i] then 
				plots(bleus[i],TRAIN_VAL_TEST[i] .. ' (BLEU)') 
				log_to_file(bleus[i],params.save_path .. '/logs/bleu/' .. TRAIN_VAL_TEST[i])
			end
			if params.log_slot_err and params.slot_err_log[i] then 
				plots(slot_errors[i],TRAIN_VAL_TEST[i] .. ' (Slot Error)') 
				log_to_file(slot_errors[i],params.save_path .. '/logs/slot/' .. TRAIN_VAL_TEST[i])
			end
		end
	end
end

run_flow()


