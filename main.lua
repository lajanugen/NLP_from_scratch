require 'nngraph'
require 'cunn'
require 'network'
require 'sll_model'
require 'wll_model'
require 'data'
require 'optim'
require 'utils'
require 'folder_utils'

--require('mobdebug').start()
--require('mobdebug').off()

params = {
task					= 'POS',
SENNA_dict				= false,
cap_feat				= false,
window_size				= 5,
vocab_size				= 100,
batch_size				= 128,
num_tags				= 1,
lr						= 0.01,
lr_decay				= false,
layers					= 3,
layer_size				= {0, 300},
objective				= 'sll',
init_with_wll			= false,
init_with_wll_path		= 'results/19/models/paramx10',
custom_optimizer		= false,
init_weight				= 0.001,
max_max_epoch			= 100,
max_epoch				= 3,
stats_freq				= 1,
log_err					= true,
log_err_freq			= 1,
err_log					= {false,true,true},
embedding_size  		= 300,
caps_embedding_size 	= 5,
use_embeddings  		= false,
embeddings_fixed		= false,
caps_feats				= true,
senna_vocab				= true,
seq_length				= 100,
use_gpu					= false,
dummy_data				= true,
A_grad					= true,
model_checkpoint		= true,
model_checkpoint_freq	= 1,
max_grad_norm			= 5,
backprop				= true,
grad_clipping			= true,
update_params			= true,

nosave					= true,
test					= false,
pre_init_mdl			= false,
pre_init_mdl_path		= 'results/23/models/',
pre_init_mdl_name		= 'paramx28',
save_results			= true,
log_dir					= true,
split					= 0.002
}

opt = {
optimizer		= 'adam',
learning_rate	= 0.001,
momentum		= 0.5
--decay_rate		= 0.9
}

if		params.task == 'Chunking'	then params.test_file = '/home/llajan/data/Chunking/test.txt'
elseif	params.task == 'NER'		then params.test_file = '/home/llajan/data/ner/eng.testb'
end

if params.test then params.pre_init_mdl = true end
if not params.cap_feat then params.caps_embedding_size = 0 end

if params.nosave or not params.test then
	folder_mgmt()
	mkdirs()
end

function transfer_data(x)
	if params.use_gpu then	return x:cuda()
	else					return x
	end
end

if params.use_gpu then g_init_gpu(arg) end
data = data()
params.layer_size[1] = params.window_size * (params.embedding_size + params.caps_embedding_size)
table.insert(params.layer_size, params.target_size)

if params.objective == 'wll' then	
	model = wll_model()
	params.custom_optimizer = true
else								
	params.batch_size = 1
	params.lr = 0.01
	model = sll_model()
end

function feval(x)
	if x ~= model.paramx then
		model.paramx:copy(x)
	end
	local pred
	if params.cap_feat then	pred = model.core_network:forward({data_x, data_x_caps})
	else					pred = model.core_network:forward(data_x)
	end

	graph.dot(model.core_network.fg, 'wll_network', 'fg')

	local err = model.criterion:forward(pred, data_y)
	local df_dw = model.criterion:backward(pred, data_y)
	model.paramdx:fill(0)
	if params.cap_feat then model.core_network:backward({data_x, data_x_caps}, df_dw)
	else					model.core_network:backward(data_x, df_dw)
	end

	return err, model.paramdx
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
	local val_errors = {}
	local test_errors = {}
	local perps, n_elems
	local perp, n_elem

	while epoch < params.max_max_epoch do
		--print(step)

		data_x, data_x_caps, data_y, done = data:get_next_batch(1,false)

		if params.custom_optimizer then
			if		opt.optimizer=='rmsprop' then	_, loss = optim.rmsprop(feval, paramx, optim_config)
			elseif	opt.optimizer=='adagrad' then	_, loss = optim.adagrad(feval, paramx, optim_config)
			elseif	opt.optimizer=='sgd'	 then	_, loss = optim.sgd(feval, model.paramx, optim_config)
			elseif	opt.optimizer=='adam'	 then 	_, loss = optim.adam(feval, model.paramx, optim_config)
			else	error('undefined optimizer')
			end
			perp = loss[1]
			--n_elem = params.batch_size
		else
			--print(step)
			--print(data_x)
			if params.cap_feat then perp = model:pass({data_x, data_x_caps}, data_y)
			else					perp = model:pass({data_x}, data_y)
			end
		end

		if perps == nil then
			perps   = torch.zeros(epoch_size):add(perp)
			--n_elems = torch.zeros(epoch_size):add(n_elem)
		end
		perps[step % epoch_size + 1] = perp
		--print(perp)
		--n_elems[step % epoch_size + 1] = n_elem

		step = step + 1
		total_cases = total_cases + params.batch_size
		--epoch = step / epoch_size
		--if step % torch.round(epoch_size * params.stats_freq) == 0 then
		if step % 100 == 0 then
			
			local wps = torch.floor(total_cases / torch.toc(start_time))
			local since_beginning = g_f3(torch.toc(beginning_time) / 60)
			--', train perp. = ' .. g_f3(torch.exp(perps:mean())) ..
			--print('word_embs',torch.sum(word_embs))
			
			local perp = perps:mean()
			print('epoch = ' .. g_f3(epoch + step/epoch_size) ..
			--', step = ' .. g_f3(step) .. 
			', train error = ' .. g_f3(perp) ..
			', wps = ' .. wps ..
			--', dw:norm() = ' .. g_f3(model.norm_dw) ..
			', lr = ' ..  g_f3(params.lr) ..
			', since beginning = ' .. since_beginning .. ' mins.')
		end

		if done or (step % epoch_size == 0) then
			table.insert(train_errors,perps:mean())
			local val  = model:run(2)
			local test = model:run(3)
			table.insert(val_errors, val)
			table.insert(test_errors, test)
			if not params.nosave then
				torch.save(params.save_path .. 'train_errors.t7', train_errors)
				torch.save(params.save_path .. 'val_errors.t7', val_errors)
				torch.save(params.save_path .. 'test_errors.t7', test_errors)
			end

			if params.lr_decay and params.objective == 'sll' then
				--params.lr = params.lr / params.decay
				params.lr = params.lr/10
				--opt.learning_rate = 0.0001
			end
			perps = nil
			if params.model_checkpoint and not params.nosave then
				torch.save(params.save_path .. 'models/paramx' .. tostring(epoch), model.paramx);
				torch.save(params.save_path .. 'models/A' .. tostring(epoch), model.A)
				torch.save(params.save_path .. 'models/A0' .. tostring(epoch), model.A0)
			end
			epoch = epoch + 1
			step = 0

			data:reset_pointer(1)
			data:reset_pointer(2)
			data:reset_pointer(3)
		end
		--if params.log_err and (step > 0) and (step % (epoch_size*params.log_err_freq) == 0) then
		--if step % 1000 == 0 then
		--	--for i = 1,3 do if params.err_log[i] then table.insert(errors[i],run(i)) end end
		--	model:run(2)
		--end

		if step % 33 == 0 then
			cutorch.synchronize()
			--collectgarbage()
		end

		--if params.model_checkpoint and (step > 0) and ((step % 100000) == 0) then
		--	torch.save(params.save_path .. 'models/paramx' .. tostring(epoch) .. '_' .. tostring(step),model.paramx);
		--	--torch.save(params.save_path .. 'models/model' .. tostring(epoch),model);
		--end
	end

	--ProFi:stop()
	--ProFi:writeReport('profiling_report')

	if params.save_models then
		torch.save(params.save_path .. 'models/model.paramx',model.paramx);
		--torch.save(params.save_path .. 'models/model',model.paramx);
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

if params.test then 
	local acc, predictions, words = model:run(3)
	local target_file = params.pre_init_mdl_path .. 'results_' .. params.pre_init_mdl_name
	if params.save_results then
		fp = io.open(target_file, 'w')
		local count = 1
		for line in io.lines(params.test_file) do
			u, v, w = unpack(string.split(line, ' '))
			if w then
				--print(u, words[count])
				--assert(u == words[count])
				line = line .. ' ' .. data.tag_invmap[predictions[count]]
				count = count + 1
			end
			fp:write(line,"\n")
		end
		fp:close()
	end
	os.execute('./chunking_eval.pl < ' .. target_file)
else	
	run_flow()
end


