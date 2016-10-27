require 'Linear_'
function gru_(x, prev_h)
  -- Calculate all four gates in one go
  local i2h = nn.Linear_(params.input_size, 2*params.rnn_size, true,  {'/home/llajan/skip-thoughts/params_txt/encoder_W','/home/llajan/skip-thoughts/params_txt/encoder_b'})(x)
  local h2h = nn.Linear_(params.rnn_size, 2*params.rnn_size, false, {'/home/llajan/skip-thoughts/params_txt/encoder_U'})(prev_h)
  local gates = nn.CAddTable()({i2h, h2h})
  local gates = nn.Sigmoid()(gates)

  local reshaped_gates	= nn.Reshape(2,params.rnn_size)(gates)
  local sliced_gates	= nn.SplitTable(2)(reshaped_gates)
  local r_gate          = nn.SelectTable(1)(sliced_gates)
  local z_gate			= nn.SelectTable(2)(sliced_gates)

  local h_x = nn.Linear_(params.input_size, params.rnn_size, true,  {'/home/llajan/skip-thoughts/params_txt/encoder_Wx','/home/llajan/skip-thoughts/params_txt/encoder_bx'})(x)
  local h_h = nn.Linear_(params.rnn_size, params.rnn_size, false, {'/home/llajan/skip-thoughts/params_txt/encoder_Ux'})(prev_h)
  local r_h = nn.CMulTable()({r_gate,h_h})
  local h_prop = nn.CAddTable()({h_x,r_h})
  local h_prop = nn.Tanh()(h_prop)

  local next_h_1 = nn.CMulTable()({z_gate, prev_h})
  local next_h_2 = nn.CMulTable()({nn.AddConstant(1)(nn.MulConstant(-1)(z_gate)),h_prop})

  local next_h = nn.CAddTable()({next_h_1, next_h_2})
  
  return next_h
end

function gru(x, prev_h)
  -- Calculate all four gates in one go
  local i2h = nn.Linear(params.rnn_size, 2*params.rnn_size)(x):annotate{name='i2h'}
  local h2h = nn.Linear(params.rnn_size, 2*params.rnn_size)(prev_h):annotate{name='h2h'}
  local gates = nn.CAddTable()({i2h, h2h})
  local gates = nn.Sigmoid()(gates)

  local reshaped_gates	= nn.Reshape(2,params.rnn_size)(gates)
  local sliced_gates	= nn.SplitTable(2)(reshaped_gates)
  local r_gate          = nn.SelectTable(1)(sliced_gates)
  local z_gate			= nn.SelectTable(2)(sliced_gates)

  local h_x = nn.Linear(params.rnn_size, params.rnn_size)(x):annotate{name='h_x'}
  local h_h = nn.Linear(params.rnn_size, params.rnn_size)(prev_h):annotate{name='h_h'}
  local r_h = nn.CMulTable()({r_gate,h_h})
  local h_prop = nn.CAddTable()({h_x,r_h})

  local next_h_1 = nn.CMulTable()({z_gate, h_prop})
  local next_h_2 = nn.CMulTable({prev_h, nn.AddConstant(1)(nn.MulConstant(-1)(z_gate))})

  local next_h = nn.CAddTable()({next_h_1, next_h_2})
  
  return next_c
end


function lstm(x, prev_c, prev_h)
  -- Calculate all four gates in one go
  local i2h = nn.Linear(params.rnn_size, 4*params.rnn_size)(x)
  local h2h = nn.Linear(params.rnn_size, 4*params.rnn_size)(prev_h)
  local gates = nn.CAddTable()({i2h, h2h})
  
  -- Reshape to (batch_size, n_gates, hid_size)
  -- Then slize the n_gates dimension, i.e dimension 2
  local reshaped_gates =  nn.Reshape(4,params.rnn_size)(gates)
  local sliced_gates = nn.SplitTable(2)(reshaped_gates)
  
  -- Use select gate to fetch each gate and apply nonlinearity
  local in_gate          = nn.Sigmoid()(nn.SelectTable(1)(sliced_gates))
  local in_transform     = nn.Tanh()(nn.SelectTable(2)(sliced_gates))
  local forget_gate      = nn.Sigmoid()(nn.SelectTable(3)(sliced_gates))
  local out_gate         = nn.Sigmoid()(nn.SelectTable(4)(sliced_gates))

  local next_c           = nn.CAddTable()({
      nn.CMulTable()({forget_gate, prev_c}),
      nn.CMulTable()({in_gate,     in_transform})
  })
  local next_h           = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})

  return next_c, next_h
end

function lstm_woinp(prev_c, prev_h)
  -- Calculate all four gates in one go
  local h2h = nn.Linear(params.rnn_size, 4*params.rnn_size)(prev_h)
  local gates = h2h
  
  -- Reshape to (batch_size, n_gates, hid_size)
  -- Then slize the n_gates dimension, i.e dimension 2
  local reshaped_gates =  nn.Reshape(4,params.rnn_size)(gates)
  local sliced_gates = nn.SplitTable(2)(reshaped_gates)
  
  -- Use select gate to fetch each gate and apply nonlinearity
  local in_gate          = nn.Sigmoid()(nn.SelectTable(1)(sliced_gates))
  local in_transform     = nn.Tanh()(nn.SelectTable(2)(sliced_gates))
  local forget_gate      = nn.Sigmoid()(nn.SelectTable(3)(sliced_gates))
  local out_gate         = nn.Sigmoid()(nn.SelectTable(4)(sliced_gates))

  local next_c           = nn.CAddTable()({
      nn.CMulTable()({forget_gate, prev_c}),
      nn.CMulTable()({in_gate,     in_transform})
  })
  local next_h           = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})

  return next_c, next_h
end

function lstm_woinp2(prev_c, prev_h) -- h input is twice rnn_size, no inputs
  -- Calculate all four gates in one go
  local h2h = nn.Linear(2*params.rnn_size, 4*params.rnn_size)(prev_h)
  local gates = h2h
  
  -- Reshape to (batch_size, n_gates, hid_size)
  -- Then slize the n_gates dimension, i.e dimension 2
  local reshaped_gates =  nn.Reshape(4,params.rnn_size)(gates)
  local sliced_gates = nn.SplitTable(2)(reshaped_gates)
  
  -- Use select gate to fetch each gate and apply nonlinearity
  local in_gate          = nn.Sigmoid()(nn.SelectTable(1)(sliced_gates))
  local in_transform     = nn.Tanh()(nn.SelectTable(2)(sliced_gates))
  local forget_gate      = nn.Sigmoid()(nn.SelectTable(3)(sliced_gates))
  local out_gate         = nn.Sigmoid()(nn.SelectTable(4)(sliced_gates))

  local next_c           = nn.CAddTable()({
      nn.CMulTable()({forget_gate, prev_c}),
      nn.CMulTable()({in_gate,     in_transform})
  })
  local next_h           = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})

  return next_c, next_h
end

function lstm_set(x, prev_c, prev_h) -- h input is twice rnn_size, there is an input
  -- Calculate all four gates in one go
  local i2h = nn.Linear(params.rnn_size, 4*params.rnn_size)(x)
  local h2h = nn.Linear(2*params.rnn_size, 4*params.rnn_size)(prev_h)
  local gates = nn.CAddTable()({i2h, h2h})
  
  -- Reshape to (batch_size, n_gates, hid_size)
  -- Then slize the n_gates dimension, i.e dimension 2
  local reshaped_gates =  nn.Reshape(4,params.rnn_size)(gates)
  local sliced_gates = nn.SplitTable(2)(reshaped_gates)
  
  -- Use select gate to fetch each gate and apply nonlinearity
  local in_gate          = nn.Sigmoid()(nn.SelectTable(1)(sliced_gates))
  local in_transform     = nn.Tanh()(nn.SelectTable(2)(sliced_gates))
  local forget_gate      = nn.Sigmoid()(nn.SelectTable(3)(sliced_gates))
  local out_gate         = nn.Sigmoid()(nn.SelectTable(4)(sliced_gates))

  local next_c           = nn.CAddTable()({
      nn.CMulTable()({forget_gate, prev_c}),
      nn.CMulTable()({in_gate,     in_transform})
  })
  local next_h           = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})

  return next_c, next_h
end

function lstm_masked(x, mask, one_m_mask, prev_c, prev_h)
  -- Calculate all four gates in one go
  local i2h = nn.Linear(params.rnn_size, 4*params.rnn_size)(x)
  local h2h = nn.Linear(params.state_size, 4*params.rnn_size)(prev_h)
  local gates = nn.CAddTable()({i2h, h2h})
  
  -- Reshape to (batch_size, n_gates, hid_size)
  -- Then slize the n_gates dimension, i.e dimension 2
  local reshaped_gates =  nn.Reshape(4,params.rnn_size)(gates)
  local sliced_gates = nn.SplitTable(2)(reshaped_gates)
  
  -- Use select gate to fetch each gate and apply nonlinearity
  local in_gate          = nn.Sigmoid()(nn.SelectTable(1)(sliced_gates))
  local in_transform     = nn.Tanh()(nn.SelectTable(2)(sliced_gates))
  local forget_gate      = nn.Sigmoid()(nn.SelectTable(3)(sliced_gates))
  local out_gate         = nn.Sigmoid()(nn.SelectTable(4)(sliced_gates))

  local mask_replicated = nn.Replicate(params.rnn_size, 2)(mask)
  local one_m_mask_rep	= nn.Replicate(params.rnn_size, 2)(one_m_mask)

  --local mask_replicated = mask
  --local one_m_mask_rep	= one_m_mask
  --local one_m_mask		= nn.CSubTable()({torch.ones(params.batch_size,params.rnn_size),mask_replicated})

  local next_c_value     = nn.CAddTable()({
      nn.CMulTable()({forget_gate, prev_c}),
      nn.CMulTable()({in_gate,     in_transform})
  })
  local next_c = nn.CAddTable()({
	  nn.CMulTable()({mask_replicated, next_c_value}),
	  nn.CMulTable()({one_m_mask_rep, prev_c})
  })

  local next_h_value     = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})
  local next_h = nn.CAddTable()({
	  nn.CMulTable()({mask_replicated, next_h_value}),
	  nn.CMulTable()({one_m_mask_rep, prev_h})
  })

  return next_c, next_h
end

function read_gate_upd(x, prev_h_table, params, prev_d)
  local i2d = nn.Linear(params.input_size, params.cond_len)(x)
  local read_input = i2d
  local i, prev_h
  for i = 1,params.layers do
	local prev_h = prev_h_table[2*i]
	local alpha  = params.alphas[i]
	local h2d = nn.Linear(params.rnn_size, params.cond_len)(prev_h)
	h2d = nn.MulConstant(alpha)(h2d)
	read_input = nn.CAddTable()({read_input, h2d})
  end
  local read_gate = nn.Sigmoid()(read_input)
  local next_d = nn.CMulTable()({prev_d,read_gate}) 
  --return read_gate, next_d
  return next_d
end

function lstm_orig_2inp(x, prev_c, prev_h, params, next_d)
  -- Calculate all four gates in one go
  local i2h = nn.Linear(params.input_size, 4*params.rnn_size)(x)
  local h2h = nn.Linear(params.rnn_size, 4*params.rnn_size)(prev_h)
  local gates = nn.CAddTable()({i2h, h2h})
  
  -- Then slize the n_gates dimension, i.e dimension 2
  local reshaped_gates =  nn.Reshape(4,params.rnn_size)(gates)
  local sliced_gates = nn.SplitTable(2)(reshaped_gates)
  
  -- Use select gate to fetch each gate and apply nonlinearity
  local in_gate          = nn.Sigmoid()(nn.SelectTable(1)(sliced_gates))
  local in_transform     = nn.Tanh()(nn.SelectTable(2)(sliced_gates))
  local forget_gate      = nn.Sigmoid()(nn.SelectTable(3)(sliced_gates))
  local out_gate         = nn.Sigmoid()(nn.SelectTable(4)(sliced_gates))

  local lin_proj = nn.Linear(params.cond_len, params.rnn_size)(next_d)
  local d_to_c = nn.Tanh()(lin_proj)

  local next_c           = nn.CAddTable()({
      nn.CMulTable()({forget_gate, prev_c}),
      nn.CMulTable()({in_gate,     in_transform}), 
	  d_to_c
  })
  local next_h           = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})

  --next_d = nn.Transpose({1,2})(next_d)

  return next_c, next_h
end

function lstm_orig_3inp(x0, x1, prev_c, prev_h, params, next_d)
  -- Calculate all four gates in one go
  local i2h0 = nn.Linear(params.input_size, 4*params.rnn_size)(x0)
  local i2h1 = nn.Linear(params.input_size, 4*params.rnn_size)(x1)
  local h2h = nn.Linear(params.rnn_size, 4*params.rnn_size)(prev_h)
  local gates = nn.CAddTable()({i2h0, i2h1, h2h})
  
  -- Then slize the n_gates dimension, i.e dimension 2
  local reshaped_gates =  nn.Reshape(4,params.rnn_size)(gates)
  local sliced_gates = nn.SplitTable(2)(reshaped_gates)
  
  -- Use select gate to fetch each gate and apply nonlinearity
  local in_gate          = nn.Sigmoid()(nn.SelectTable(1)(sliced_gates))
  local in_transform     = nn.Tanh()(nn.SelectTable(2)(sliced_gates))
  local forget_gate      = nn.Sigmoid()(nn.SelectTable(3)(sliced_gates))
  local out_gate         = nn.Sigmoid()(nn.SelectTable(4)(sliced_gates))

  local lin_proj = nn.Linear(params.cond_len, params.rnn_size)(next_d)
  local d_to_c = nn.Tanh()(lin_proj)

  local next_c           = nn.CAddTable()({
      nn.CMulTable()({forget_gate, prev_c}),
      nn.CMulTable()({in_gate,     in_transform}), 
	  d_to_c
  })
  local next_h           = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})

  --next_d = nn.Transpose({1,2})(next_d)

  return next_c, next_h
end

