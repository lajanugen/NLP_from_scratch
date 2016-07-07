
local data = torch.class('data')

function tost(i)
	local i_str = ''
	if i < 10 then i_str = '0' end
	i_str = i_str .. tostring(i)
	return i_str
end

function data:read_data() 
	data_path = '/home/llajan/data/treebank/TAGGED/POS/WSJ/'
	tvt = 1
	local words = {}
	local tagset = {}
	local tagcount = 0
	local sentences = {{},{},{}}
	local tags = {{},{},{}}

	if params.task == 'POS' then
		local ul 
		if params.dummy_data then	ul = 10
		else						ul = 99
		end
		for i = 0,24 do
			if i == 19 then tvt = tvt + 1 end
			if i == 22 then tvt = tvt + 1 end
			local i_str = tost(i)
			for j = 0,ul do
				if i > 0 or j > 0 then local j_str = tost(j) local file_path = data_path .. i_str .. '/WSJ_' .. i_str .. j_str .. '.POS'
					local sent_mid = false
					local sentence = {}
					local tag = {}
					for line in io.lines(file_path) do
						if not sent_mid then
							if line ~= '' and string.sub(line,1,1) ~= '=' then sent_mid = true end
						end
						if sent_mid then
							if line == '' or string.sub(line,1,1) == '=' then 
								sent_mid = false
								--print(sentence)
								assert(#sentence > 0)
								--if #sentence == 1 then print(file_path); print(sentence) end
								table.insert(sentences[tvt],sentence)
								table.insert(tags[tvt],tag)
								sentence = {}
								tag = {}
							else
								--if string.sub(line,1,1) == '[' and string.sub(line,-1,-1) == ']' then line = string.sub(line,2,-2) end
								--if string.sub(line,1,1) == '[' and string.sub(line,-2,-2) == ']' then line = string.sub(line,2,-3) end
								local tokens = string.split(line,' ')
								for k = 1,#tokens do 
									local token = string.split(tokens[k],'/')
									if token[2] ~= nil then
										--if token[1] == nil then
										--	print(tokens)
										--end
										local word = token[1]
										if words[word] == nil then 	words[word] = 0
										else						words[word] = words[word] + 1
										end
										local tg = token[#token]
										if string.find(tg,'|') then
											tg = string.split(tg,'|')[1]
										end
										if tagset[tg] == nil then 
											tagcount = tagcount + 1
											if tg == nil then
												print(i_str,j_str)
												print(tokens)
											end
											tagset[tg] = tagcount
										end
										table.insert(sentence,word)
										table.insert(tag,tg)
									end
								end
							end
						end
					end
					--print(sentence)
					if #sentence > 0 then
						--if #sentence == 1 then print(file_path); print(sentence) end
						table.insert(sentences[tvt],sentence)
						table.insert(tags[tvt],tag)
						sentence = {}
						tag = {}
					end
				end
			end
		end
	elseif params.task == 'Chunking' then
		train_data_path = '/home/llajan/data/Chunking/train.txt'
		test_data_path	= '/home/llajan/data/Chunking/test.txt'
		local sentence, tag = {}, {}
		for _, path in ipairs({train_data_path, test_data_path}) do
			for line in io.lines(path) do
				local word, pos, tg, test = unpack(string.split(line, ' '))
				assert(test == nil)
				if word ~= nil then
					if words[word] == nil then words[word] = 0
					else words[word] = words[word] + 1
					end
				end
				if tg == nil then -- eos
					table.insert(sentences[tvt], sentence)
					table.insert(tags[tvt], tag)
					sentence = {}
					tag = {}
				else
					word = string.gsub(word, "%d+", 'NUMBER')
					table.insert(sentence, word)
					table.insert(tag, tg)
					if tagset[tg] == nil then 
						tagcount = tagcount + 1
						tagset[tg] = tagcount
					end
				end
			end
			tvt = 3
		end
		local ntrain = #sentences[1]
		local num_valid = torch.round(params.split*ntrain)
		for i = ntrain - num_valid, ntrain do
			table.insert(sentences[2], sentences[1][i])
			table.insert(tags[2], tags[1][i])
			sentences[1][i] = nil
			tags[1][i] = nil
		end
	elseif params.task == 'NER' then
		train_data_path = '/home/llajan/data/ner/eng.train'
		test_data_path	= '/home/llajan/data/ner/eng.testb'
		local sentence, tag = {}, {}
		for _, path in ipairs({train_data_path, test_data_path}) do
			ct = 0
			for line in io.lines(path) do
				ct = ct + 1
				local word, pos, _, tg, test = unpack(string.split(line, ' '))
				assert(test == nil)
				if word ~= nil then
					word = string.gsub(word, "%d+", 'NUMBER')
					word = word:lower()
					if words[word] == nil then words[word] = 0
					else words[word] = words[word] + 1
					end
				end
				if tg == nil then -- eos
					if #sentence == 0 then print(ct) end
					table.insert(sentences[tvt], sentence)
					table.insert(tags[tvt], tag)
					sentence = {}
					tag = {}
				else
					table.insert(sentence, word)
					table.insert(tag, tg)
					if tagset[tg] == nil then 
						tagcount = tagcount + 1
						tagset[tg] = tagcount
					end
				end
			end
			tvt = 3
		end
		local ntrain = #sentences[1]
		local num_valid = torch.round(params.split*ntrain)
		for i = ntrain - num_valid, ntrain do
			table.insert(sentences[2], sentences[1][i])
			table.insert(tags[2], tags[1][i])
			sentences[1][i] = nil
			tags[1][i] = nil
		end
	end
	return {sentences, tags, words, tagset, tagcount}
end			

function data:__init()
	local sentences, tags, words, tagset, tagcount = unpack(self:read_data())
	--print(tagset)
	local word_freqs = {}
	--for w,c in pairs(words) do table.insert(word_freqs,c) end
	--table.sort(word_freqs)
	--print(word_freqs)
	--word_freqs_sorted = table.sort(word_freqs)
	--print('num_words',#word_freqs)
	--local MOST_FREQ = 100000
	--local cutoff = #word_freqs - MOST_FREQ
	--print(cutoff,#word_freqs)
	--print(word_freqs[cutoff])
	


	params.target_size = tagcount
	params.num_tags = params.target_size

	DUMMY = 1
	self.vocab_map = {}
	self.ivocab_map = {}
	local word_id = 1

	if params.SENNA_dict then
		for line in io.lines('words.lst') do
			word_id = word_id + 1
			self.vocab_map[line] = word_id
			--self.ivocab_map[word_id] = word
		end
		word_id = word_id + 1
		self.vocab_map['RARE'] = word_id + 1
	else
		for word,_ in pairs(words) do 
			word = word:lower()
			if self.vocab_map[word] == nil then
				word_id = word_id + 1
				self.vocab_map[word] = word_id
				self.ivocab_map[word_id] = word
			end
		end
	end

	params.vocab_size = word_id
	self.tag_map = tagset
	self.tag_invmap = {}
	for u, v in pairs(self.tag_map) do
		self.tag_invmap[v] = u
	end

	local task_vocab_map = {}
	local task_vocab_size = 0
	--print(tagset)
	--print(tagcount)

	local sentences_caps = {{},{},{}}
	for i = 1,3 do
		for j = 1,#sentences[i] do
			local sent = sentences[i][j]
			local sent_t = {}
			local sent_c = {}
			for k = 1,(params.window_size-1)/2 do table.insert(sent_t,DUMMY) end
			for k = 1,(params.window_size-1)/2 do table.insert(sent_c,5) end
			for _,Word in ipairs(sent) do
				word = Word:lower()
				if self.vocab_map[word] ~= nil then	table.insert(sent_t,self.vocab_map[word]) 
				else								table.insert(sent_t,self.vocab_map['RARE']) 
				end
				if params.cap_feat then
					local cap_feat
					if		word == Word									then	cap_feat = 1 -- all lower
					elseif	string.sub(Word,2,-1) == string.sub(word,2,-1)	then	cap_feat = 2 -- First letter upper
					elseif  word:upper() == Word							then	cap_feat = 3 -- All upper
					else															cap_feat = 4 -- Non initial upper
					end
					table.insert(sent_c, cap_feat)
				end
			end
			for k = 1,(params.window_size-1)/2 do table.insert(sent_t,DUMMY) end
			for k = 1,(params.window_size-1)/2 do table.insert(sent_c,5) end
			--sentences[i][j] = transfer_data(torch.Tensor(sent_t))
			sentences[i][j] = torch.Tensor(sent_t)
			sentences_caps[i][j] = torch.Tensor(sent_c)

			if sentences[i][j]:size(1) == 4 then print('sent', sent, sent_t) end

			local tag_t = {}
			local tag = tags[i][j]
			for _,tg in ipairs(tag) do table.insert(tag_t,self.tag_map[tg]) end
			--tags[i][j] = transfer_data(torch.Tensor(tag_t))
			tags[i][j] = torch.Tensor(tag_t)
		end
	end

	self.sent_ptr = {1,1,1}
	self.token_ptr = {1,1,1}
	self.batch_count = {}

	self.sentences = sentences
	self.sentences_caps = sentences_caps
	self.tags = tags

	self.Nsentences = {#sentences[1],#sentences[2],#sentences[3]}
	print('sent',self.Nsentences)
end

function data:get_batch_count(tvt)
	if self.batch_count[tvt] ~= nil then
		return self.batch_count[tvt]
	else
		if params.objective == 'wll' then
			local num_data = 0
			for i = 1,self.Nsentences[tvt] do
				num_data = num_data + self.sentences[tvt][i]:size(1) - (params.window_size-1)
			end
			self.batch_count[tvt] = torch.ceil(num_data/params.batch_size)
		else
			self.batch_count[tvt] = torch.ceil(self.Nsentences[tvt]/params.batch_size)
		end
		return self.batch_count[tvt]
	end
end

function data:get_next_batch(tvt,rand)
	local sentences = self.sentences[tvt]
	local sentences_caps = self.sentences_caps[tvt]
	local targets = self.tags[tvt]
	local Nsent = #self.sentences[tvt]

	local batch
	local batch_caps
	local batch_target
	local done = false

	if params.objective == 'wll' then
		batch			= transfer_data(torch.ones(params.batch_size,params.window_size))
		if params.cap_feat then
			batch_caps		= transfer_data(torch.ones(params.batch_size,params.window_size))
		end
		batch_target	= transfer_data(torch.ones(params.batch_size))
		local x,y
		for i = 1,params.batch_size do
			if rand then
				x = torch.random(Nsent)
				if (sentences[x]:size(1) <= params.window_size-1) then print(sentences[x]) end
				y = torch.random(sentences[x]:size(1) - (params.window_size-1))
			else
				if self.token_ptr[tvt] > sentences[self.sent_ptr[tvt]]:size(1) - (params.window_size-1) then
					self.sent_ptr[tvt] = self.sent_ptr[tvt] + 1
					self.token_ptr[tvt] = 1
				end
				if self.sent_ptr[tvt] > self.Nsentences[tvt] then 
					self.sent_ptr[tvt] = 1
					self.token_ptr[tvt] = 1
				end
				x = self.sent_ptr[tvt]
				y = self.token_ptr[tvt]
				self.token_ptr[tvt] = self.token_ptr[tvt] + 1
			end
			assert(sentences[x]:size(1) >= y+params.window_size-1)
			batch[i] = sentences[x]:sub(y,y+params.window_size-1)
			if params.cap_feat then
				batch_caps[i] = sentences_caps[x]:sub(y,y+params.window_size-1)
			end
			batch_target[i] = targets[x][y]
		end
	else

		--if sentences[self.sent_ptr[tvt]]:size(1) < params.window_size + 1 then 
		--	self.sent_ptr[tvt] = self.sent_ptr[tvt] + 1
		--end

		batch, batch_caps = self:next_batch_helper(tvt)

		batch_target	= transfer_data(targets[self.sent_ptr[tvt]])
		if not params.dummy_data then
			self.sent_ptr[tvt] = self.sent_ptr[tvt] + 1
		end
		if self.sent_ptr[tvt] > self.Nsentences[tvt] then 
			self.sent_ptr[tvt] = 1
			done = true
		end
	end
	return batch, batch_caps, batch_target, done
end

function data:next_batch_helper(tvt)
	batch			= transfer_data(self.sentences[tvt][self.sent_ptr[tvt]])
	if params.cap_feat then
		batch_caps		= transfer_data(self.sentences_caps[tvt][self.sent_ptr[tvt]])
	end
	if batch:dim() == 1 then
		batch:resize(1,batch:size(1))
		if params.cap_feat then batch_caps:resize(1,batch_caps:size(1)) end
	elseif batch:size(2) == 1 then
		batch = batch:t()
		if params.cap_feat then batch_caps = batch_caps:t() end
	end
	while batch:size(2) < params.window_size + 1 do 
		self.sent_ptr[tvt] = self.sent_ptr[tvt] + 1
		--print(batch:size())
		--print(batch)
		print('SKIP')
		return self:next_batch_helper(tvt)
	end
	return batch, batch_caps
end

function data:reset_pointer(tvt)
	self.sent_ptr[tvt] = 1
	self.token_ptr[tvt] = 1
end
