# define training encoder
inf_enc_out, inf_enc_state = tf.nn.dynamic_rnn(
    enc_cell,
    inf_enc_inp,
    dtype = tf.float32,
    sequence_length = seq_length_inp,
    time_major = time_major)

with tf.variable_scope('projection_layer', reuse = tf.AUTO_REUSE):
    from tensorflow.python.layers.core import Dense
    projection_layer = Dense(features_dec_exp_out)

# define inference custom helper
def initialize_fn():
    finished = tf.tile([False], [batch_size])
    enc_inp_end = inf_enc_inp[0, observation_length - 1, 0]
    start_inputs = tf.reshape(enc_inp_end, shape=[1, 1])
    return (finished, start_inputs)

def sample_fn(time, outputs, state):
    return tf.identity([batch_size])

def next_inputs_fn(time, outputs, state, sample_ids):
    finished = time >= prediction_length
    next_inputs = outputs
    return (finished, next_inputs, state)

inf_custom_helper = tf.contrib.seq2seq.CustomHelper(
    initialize_fn = initialize_fn,
    sample_fn = sample_fn,
    next_inputs_fn = next_inputs_fn)

# create inference decoder
inf_decoder = tf.contrib.seq2seq.BasicDecoder(
    dec_cell,
    inf_custom_helper,
    inf_enc_state,
    projection_layer)

# create inference dynamic decoding
inf_dec_out, inf_dec_state, inf_dec_out_seq_length = tf.contrib.seq2seq.dynamic_decode(
    inf_decoder,
    output_time_major = time_major)

# extract prediction from decoder output
inf_output_dense = inf_dec_out.rnn_output