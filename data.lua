
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
	local ul 
	if params.dummy_data then ul = 10
	else ul = 99
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
									if words[token[1]] == nil then 	words[token[1]] = 0
									else							words[token[1]] = words[token[1]] + 1
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
									table.insert(sentence,token[1])
									table.insert(tag,tg)
								end
							end
						end
					end
				end
				--print(sentence)
				if #sentence > 0 then
					table.insert(sentences[tvt],sentence)
					table.insert(tags[tvt],tag)
					sentence = {}
					tag = {}
				end
			end
		end
	end
	return {sentences, tags, words, tagset, tagcount}
end			

function data:__init()
	local sentences, tags, words, tagset, tagcount = unpack(self:read_data())
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
	local word_id = 1
	for word,_ in pairs(words) do 
		if self.vocab_map[word] == nil then
			word_id = word_id + 1
			self.vocab_map[word] = word_id
		end
	end
	params.vocab_size = word_id
	self.tag_map = tagset

	--print(tagset)
	--print(tagcount)

	for i = 1,3 do
		for j = 1,#sentences[i] do
			local sent = sentences[i][j]
			local sent_t = {}
			for k = 1,(params.window_size-1)/2 do table.insert(sent_t,DUMMY) end
			for _,word in ipairs(sent) do table.insert(sent_t,self.vocab_map[word]) end
			for k = 1,(params.window_size-1)/2 do table.insert(sent_t,DUMMY) end
			sentences[i][j] = transfer_data(torch.Tensor(sent_t))

			if sentences[i][j]:size(1) == 4 then print('sent', sent) end

			local tag_t = {}
			local tag = tags[i][j]
			for _,tg in ipairs(tag) do table.insert(tag_t,self.tag_map[tg]) end
			tags[i][j] = transfer_data(torch.Tensor(tag_t))
		end
	end

	self.sent_ptr = {1,1,1}
	self.token_ptr = {1,1,1}
	self.batch_count = {}

	self.sentences = sentences
	self.tags = tags

	self.Nsentences = {#sentences[1],#sentences[2],#sentences[3]}
end

function data:get_batch_count(tvt)
	if self.batch_count[tvt] ~= nil then
		return self.batch_count[tvt]
	else
		local num_data = 0
		for i = 1,self.Nsentences[tvt] do
			num_data = num_data + self.sentences[tvt][i]:size(1) - (params.window_size-1)
		end
		self.batch_count[tvt] = torch.round(num_data/params.batch_size)
		return self.batch_count[tvt]
	end
end

function data:get_next_batch(tvt,rand)
	local sentences = self.sentences[tvt]
	local targets = self.tags[tvt]
	local Nsent = #self.sentences[tvt]

	local batch
	local batch_target

	if params.objective == 'wll' then
		batch			= transfer_data(torch.ones(params.batch_size,params.window_size))
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
			batch_target[i] = targets[x][y]
		end
	else
		if self.sent_ptr[tvt] > self.Nsentences[tvt] then 
			self.sent_ptr[tvt] = 1
		end
	
		--while sentences[self.sent_ptr[tvt]]:size(1) < params.window_size + 1 do 
		--	self.sent_ptr[tvt] = self.sent_ptr[tvt] + 1
		--end

		batch			= transfer_data(sentences[self.sent_ptr[tvt]])
		if batch:dim() == 1 then
			batch:resize(1,batch:size(1))
		end
		batch_target	= transfer_data(targets[self.sent_ptr[tvt]])
		if not params.dummy_data then
			self.sent_ptr[tvt] = self.sent_ptr[tvt] + 1
		end
	end
	return batch, batch_target
end
