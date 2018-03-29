from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# We disable pylint because we need python3 compatibility.
from six.moves import xrange  # pylint: disable=redefined-builtin
# from tensorflow.contrib.rnn.python.ops import core_rnn_cell_impl
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest
import tensorflow as tf
import numpy as np

# TODO(ebrevdo): Remove once _linear is fully deprecated.
linear = rnn_cell_impl._linear  # pylint: disable=protected-access


def attention_encoder(encoder_inputs, prev_y, attention_states, cell,
                      output_size=None, num_heads=1,
                      dtype=dtypes.float32, scope=None):
    """RNN encoder with attention.
  In this context "attention" means that, during encoding, the RNN can look up
  information in the additional tensor "attention_states", which is constructed by transpose the dimensions of time steps and input features of the inputs,
  and it does this to focus on a few features of the input.

  Args:
    encoder_inputs: A list of 2D Tensors [batch_size x n_input_encoder].
    initial_state: 2D Tensor [batch_size x cell.state_size].
    attention_states: 3D Tensor [batch_size x attn_length x attn_size].
    cell: rnn_cell.RNNCell defining the cell function and size.
    output_size: Size of the output vectors; if None, we use cell.output_size.
    num_heads: Number of attention heads that read from attention_states.
    dtype: The dtype to use for the RNN initial state (default: tf.float32).
    scope: VariableScope for the created subgraph; default: "attention_decoder".

  Returns:
    A tuple of the form (outputs, state, attn_weights), where:
      outputs: A list of the encoder hidden states. Each element is a 2D Tensor of shape [batch_size x output_size].
      state: The state of encoder cell at the final time-step. It is a 2D Tensor of shape [batch_size x cell.state_size].
      attn_weights: A list of the input attention weights. Each element is a 2D Tensor of shape [batch_size x attn_length]
  Raises:
    ValueError: when num_heads is not positive, there are no inputs, shapes
      of attention_states are not set, or input size cannot be inferred
      from the input.
  """
    if not encoder_inputs:
        raise ValueError("Must provide at least 1 input to attention encoder.")
    if num_heads < 1:
        raise ValueError("With less than 1 heads, use a non-attention encoder.")
    if not attention_states.get_shape()[1:2].is_fully_defined():
        raise ValueError("Shape[1] and [2] of attention_states must be known: %s"
                         % attention_states.get_shape())
    if output_size is None:
        output_size = cell.output_size

    with variable_scope.variable_scope(scope or "attention_encoder"):
        # get the batch_size of the encoder_input
        batch_size = array_ops.shape(encoder_inputs[0])[0]  # Needed for reshaping.
        # attention_state.shape (batch_size, n_input_encoder, n_steps_encoder)
        attn_length = attention_states.get_shape()[1].value  # n_input_encoder
        attn_size = attention_states.get_shape()[2].value  # n_steps_encoder

        # To calculate W1 * h_t we use a 1-by-1 convolution, need to reshape before.
        # hidden_features shape: (batch_size, attn_length, 1, attn_size)
        hidden = array_ops.reshape(
            attention_states, [-1, attn_length, 1, attn_size])
        # Size of query vectors for attention.

        # k = variable_scope.get_variable("Attn_EncoderW" ,
        #                              [1, 1, attn_size, attn_size])
        # hidden_features=nn_ops.conv2d(hidden, k, [1, 1, 1, 1], "SAME")
        # hidden_features=tf.transpose(hidden,[0,3,2,1])
        # k_1 = variable_scope.get_variable("Attn_EncoderW_1" ,
        #                              [1, 1, attn_length, attn_length])
        # [1, 1, attn_length, 1])
        # hidden_features_1=hidden_features
        # hidden_features_1=nn_ops.conv2d(hidden_features, k_1, [1, 1, 1, 1], "SAME")
        # k_2 = variable_scope.get_variable("Attn_EncoderW_3" ,
        #                              [attn_length, 1, attn_size, attention_vec_size])
        # hidden_features_2=nn_ops.conv2d(hidden, k_2, [1, 1, 1, 1], "SAME")
        v = variable_scope.get_variable("AttnEncoderV_1", [attn_length])
        v_1 = variable_scope.get_variable("AttnEncoderV_2", [attn_length])
        v_2 = variable_scope.get_variable("AttnEncoderV_3", [attn_length])
        # v_3 = variable_scope.get_variable("AttnEncoderV_4",[attn_length])
        # v_4 = variable_scope.get_variable("AttnEncoderV_5",[attn_length])
        # how to get the initial_state
        # initial_state_size = array_ops.stack()
        initial_state = [array_ops.zeros([batch_size, output_size], dtype=dtype) for _ in xrange(2)]
        state = initial_state

        def attention(query, i, attns):
            """Put attention masks on hidden using hidden_features and query."""
            if nest.is_sequence(query):  # If the query is a tuple, flatten it.
                query_list = nest.flatten(query)
                for q in query_list:  # Check that ndims == 2 if specified.
                    ndims = q.get_shape().ndims
                    if ndims:
                        assert ndims == 2
                query = array_ops.concat(query_list, 1)

            with variable_scope.variable_scope("AttentionEncoder"):
                # y with the shape (batch_size, attention_vec_size)
                y = linear(query, attn_length, True)
                # y with the shape (batch_size, 1, 1, attention_vec_size)
                y = array_ops.reshape(y, [-1, 1, attn_length])
            with variable_scope.variable_scope("conv2"):
                y_1 = linear(query, attn_length, True)
                y_1 = array_ops.reshape(y_1, [-1, attn_length, 1])
                attns = array_ops.reshape(attns, [-1, attn_length, 1])
                # incs =array_ops.reshape(attns,[-1,attn_length,1])
                # Attention mask is a softmax of v^T * tanh(...).
                # hidden_features with the shape (batch_size, attn_length, 1,attn_size )
                # s = math_ops.reduce_sum(v * math_ops.tanh(hidden_features + y), [2, 3])
                # s = tf.reduce_sum(v * tf.nn.tanh(hidden[:,:,:,i] + y) +v_2*tf.nn.tanh(attns), [2])
                # s = tf.reduce_sum(v * tf.nn.tanh(hidden[:,:,:,i]) +v_2*tf.nn.tanh(attns)+v_3*tf.nn.tanh(y), [2])
                s = tf.reduce_sum(v * tf.nn.tanh(hidden[:, :, :, i] + y + v_2 * attns), [2])
                # a with shape (batch_size, attn_length)
                # a is the attention weight
                # a = nn_ops.softmax(s)
                # a=s
                a = tf.nn.softmax(s)
                # b = math_ops.reduce_sum(v_1 * math_ops.tanh(hidden_features_1 + y_1), [1, 3])
                b = tf.reduce_sum(v_1 * tf.nn.tanh(hidden[:, :, :, i] + y_1), [2])  # +v_3*tf.nn.tanh(incs)
                # beta= tf.nn.softmax(b)
                # b = tf.reduce_sum(v_1 * tf.nn.tanh(hidden_features_1[:,i,:,:] ), [1]) +v_3*tf.nn.tanh(incs)  #no
                # b = tf.reduce_sum(v_1 * tf.nn.tanh(hidden[:,:,:,i]+ incs ), [2]) #no, have to have y

            return a, b

        outputs = []
        attn_weights = []
        intercepts = []
        # batch_attn_size = array_ops.stack()
        # batch_attn_size_1 = array_ops.stack()
        attns = array_ops.zeros([batch_size, attn_length], dtype=dtype)
        incs = array_ops.zeros([batch_size, attn_length], dtype=dtype)
        # es = array_ops.zeros([batch_size, attn_length], dtype=dtype)
        # i is the index of the which time step
        # inp is numpy.array and the shape of inp is (batch_size, n_feature)
        for i, inp in enumerate(encoder_inputs):
            if i > 0:
                variable_scope.get_variable_scope().reuse_variables()
            input_size = inp.get_shape().with_rank(2)[1]
            if input_size.value is None:
                raise ValueError("Could not infer input size from input: %s" % inp.name)

            # multiply attention weights with the original input
            # get the newly input
            inputs = attns * inp
            # mean,var= tf.nn.moments(inp,axes=[1])
            # var_term=mean*var
            # var=tf.reshape(var,[-1,1])
            # var_term=tf.reshape(var_term,[-1,1])

            # x = array_ops.concat([inputs,prev_y[i],incs,var_term],1)
            # x = array_ops.concat([inputs,prev_y[i],incs],1)#best?
            # x = array_ops.concat([inputs,prev_y[i],var_term],1)
            # x = array_ops.concat([inputs,incs,var_term],1)
            x = array_ops.concat([inputs, incs], 1)
            # Run the BasicLSTM with the newly input
            cell_output, state = cell(x, state)

            # Run the attention mechanism.
            attns, incs = attention(state, i, attns)

            with variable_scope.variable_scope("AttnEncoderOutputProjection"):
                output = cell_output

            outputs.append(output)
            attn_weights.append(attns)
            intercepts.append(incs)

    return outputs, state, attn_weights, intercepts

