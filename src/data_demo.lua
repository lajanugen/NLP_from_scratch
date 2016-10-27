local data = torch.class('data')

function tost(i)
	local i_str = ''
	if i < 10 then i_str = '0' end
	i_str = i_str .. tostring(i)
	return i_str
end

function data:read_data() 
	--data_path = '/z/data/treebank_text/'
	data_path = '/home/llajan/data/treebank/TAGGED/POS/WSJ/'
	tvt = 1
	local words = {}
	--local tagset = {}
	local tagcount = 0
	local tagset = torch.load(params.task .. '_tagset')
	local itagset = {}
	for u,v in pairs(tagset) do 
		itagset[v] = u
		tagcount = tagcount + 1 
	end
	self.itagset = itagset
	local sentences = {{},{},{}}
	local tags = {{},{},{}}

	return sentences, tags, words, tagset, tagcount
end			

function data:__init()
	local sentences, tags, words, tagset, tagcount = self:read_data()
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
	word_id = word_id + 1
	self.vocab_map['RARE'] = word_id

	params.vocab_size = word_id
	self.tag_map = tagset
	self.tag_invmap = {}
	for u, v in pairs(self.tag_map) do
		self.tag_invmap[v] = u
	end

	local task_vocab_map = {}
	local task_vocab_size = 0
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
			--print(i,x,y,targets[x])
			--print(targets)
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

function data:proc(sentence)
	sent = string.split(sentence, ' ')

	local sent_t = {}
	local sent_c = {}
	for k = 1,(params.window_size-1)/2 do table.insert(sent_t,DUMMY) end
	for k = 1,(params.window_size-1)/2 do table.insert(sent_c,5) end
	for _, Word in ipairs(sent) do
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
	sent_t = transfer_data(torch.Tensor(sent_t))--:view(1, #sent_t)
	sent_c = transfer_data(torch.Tensor(sent_c))--:view(1, #sent_c)
	return sent_t, sent_c, sent
end
