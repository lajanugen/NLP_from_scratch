function plots(errors,title)
	num_data = #errors
	local x = torch.linspace(1,num_data,num_data)
	local plot = {}
	gnuplot.title(title)
	gnuplot.xlabel('Epoch')
	--gnuplot.ylabel(ylabel)
	gnuplot.raw('set terminal pngcairo')
	gnuplot.raw('set pointsize 0')
	gnuplot.raw('set output "' .. params.save_path .. '/plots/' .. title .. '.png"')
	gnuplot.plot({x,torch.Tensor(errors)})
	
end

function log_to_file(data,file)
	f = io.open(file,'w')
	for i = 1,#data do
		f:write(data[i])
		if i ~= #data then
			f:write('\n')
		end
	end
end

function folder_mgmt()
-- Folder management --
	local description_file = '../results/descriptions'
	local descriptions_read = io.open(description_file, 'r')
	local descriptions_write = io.open(description_file, 'a')
	local last_line
	for line in descriptions_read:lines() do if line then last_line = line end end
	line = string.split(last_line," ")
	if params.desc then
		if params.desc == '0' then
			results_number = line[1]
		else
			results_number = tostring(tonumber(line[1]) + 1)
			descriptions_write:write(results_number .. ' ' .. params.desc .. '\n')
		end
	else
		error("Enter description")
	end
	params.save_path = '../results/' .. results_number .. '/'
	descriptions_read:close()
	descriptions_write:close()
	print('Directory done')
end

function mkdirs()
	if params.log_dir then
		local save_path = params.save_path
		print(params.save_path)
		os.execute("mkdir " .. save_path)
		os.execute("mkdir " .. save_path .. 'models')
		--os.execute("mkdir -p " .. save_path .. 'train/gen')
		--os.execute("mkdir -p " .. save_path .. 'train/bleu')
		--os.execute("mkdir -p " .. save_path .. 'train/bleu_eval')
		--os.execute("mkdir -p " .. save_path .. 'train/bleu_eval_scores')
		--os.execute("mkdir -p " .. save_path .. 'val/gen')
		--os.execute("mkdir -p " .. save_path .. 'val/bleu')
		--os.execute("mkdir -p " .. save_path .. 'val/bleu_eval')
		--os.execute("mkdir -p " .. save_path .. 'val/bleu_eval_scores')
		--os.execute("mkdir -p " .. save_path .. 'test/gen')
		--os.execute("mkdir -p " .. save_path .. 'test/bleu')
		--os.execute("mkdir -p " .. save_path .. 'test/bleu_eval')
		--os.execute("mkdir -p " .. save_path .. 'test/bleu_eval_scores')
		--os.execute("mkdir -p " .. save_path .. 'test/bleu_eval_scores')
		--os.execute("mkdir -p " .. save_path .. 'plots')
		--os.execute("mkdir -p " .. save_path .. 'logs/err')
		--os.execute("mkdir -p " .. save_path .. 'logs/bleu')
		--os.execute("mkdir -p " .. save_path .. 'logs/slot')
		if params.save_embeddings then
			os.execute("mkdir -p " .. save_path .. 'embeddings')
		end

		param_file = io.open(save_path .. "params","w")
		for key, value in pairs (params) do
		    param_file:write(string.format("[%s] : %s\n",
		    tostring (key), tostring(value)))
		end
		param_file:close()
	end
end
