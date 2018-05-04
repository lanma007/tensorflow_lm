from __future__ import print_function

import collections

import tensorflow as tf

__all__ = ["BatchedInput", "get_iterator", "get_infer_iterator"]


# NOTE(ebrevdo): When we subclass this, instances' __dict__ becomes empty.
class BatchedInput(
    collections.namedtuple("BatchedInput",
                           ("initializer", "source", "target_input",
                            "target_output", "source_sequence_length",
                            "target_sequence_length"))):
  pass


def get_infer_iterator(src_dataset,
                       batch_size,
                       src_max_len=None):
  #src_eos_id = tf.cast(src_vocab_table.lookup(tf.constant(eos)), tf.int32)
  #src_dataset = src_dataset.map(lambda src: tf.string_split([src]).values)

  if src_max_len:
    src_dataset = src_dataset.map(lambda src: src[:src_max_len])
  # Convert the word strings to ids
  #src_dataset = src_dataset.map(
  #    lambda src: tf.cast(src_vocab_table.lookup(src), tf.int32))
  # Add in the word counts.
  src_dataset = src_dataset.map(lambda src: (src, tf.size(src)))

  def batching_func(x):
    return x.batch(batch_size)

  batched_dataset = batching_func(src_dataset)
  batched_iter = batched_dataset.make_initializable_iterator()
  (src_ids, src_seq_len) = batched_iter.get_next()
  return BatchedInput(
      initializer=batched_iter.initializer,
      source=src_ids,
      target_input=None,
      target_output=None,
      source_sequence_length=src_seq_len,
      target_sequence_length=None)


def get_iterator(src_dataset,
                 tgt_dataset,
                 # src_vocab_table,
                 # tgt_vocab_table,
                 batch_size,
                 # sos,
                 # eos,
                 random_seed,
                 # num_buckets,
                 # src_max_len=None,
                 # tgt_max_len=None,
                 num_parallel_calls=4,
                 output_buffer_size=None,
                 skip_count=None,
                 num_shards=1,
                 shard_index=0,
                 reshuffle_each_iteration=True):
  if not output_buffer_size:
   output_buffer_size = batch_size * 1000
  # src_eos_id = tf.cast(src_vocab_table.lookup(tf.constant(eos)), tf.int32)
  # tgt_sos_id = tf.cast(tgt_vocab_table.lookup(tf.constant(sos)), tf.int32)
  # tgt_eos_id = tf.cast(tgt_vocab_table.lookup(tf.constant(eos)), tf.int32)

  src_tgt_dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset))

  src_tgt_dataset = src_tgt_dataset.shard(num_shards, shard_index)
  if skip_count is not None:
    src_tgt_dataset = src_tgt_dataset.skip(skip_count)

  src_tgt_dataset = src_tgt_dataset.shuffle(
      output_buffer_size, random_seed, reshuffle_each_iteration)

  # src_tgt_dataset = src_tgt_dataset.map(
  #     lambda src, tgt: (
  #         tf.string_split([src]).values, tf.string_split([tgt]).values),
  #     num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

  # Filter zero length input sequences.
  # src_tgt_dataset = src_tgt_dataset.filter(
  #     lambda src, tgt: tf.logical_and(tf.size(src) > 0, tf.size(tgt) > 0))

  # if src_max_len:
  #   src_tgt_dataset = src_tgt_dataset.map(
  #       lambda src, tgt: (src[:src_max_len], tgt),
  #       num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
  # if tgt_max_len:
  #   src_tgt_dataset = src_tgt_dataset.map(
  #       lambda src, tgt: (src, tgt[:tgt_max_len]),
  #       num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
  # Convert the word strings to ids.  Word strings that are not in the
  # vocab get the lookup table's default_value integer.
  # src_tgt_dataset = src_tgt_dataset.map(
  #     lambda src, tgt: (tf.cast(src_vocab_table.lookup(src), tf.int32),
  #                       tf.cast(tgt_vocab_table.lookup(tgt), tf.int32)),
  #     num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
  # # Create a tgt_input prefixed with <sos> and a tgt_output suffixed with <eos>.
  # src_tgt_dataset = src_tgt_dataset.map(
  #     lambda src, tgt: (src,
  #                       tf.concat(([tgt_sos_id], tgt), 0),
  #                       tf.concat((tgt, [tgt_eos_id]), 0)),
  #     num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
  # # Add in sequence lengths.
  src_tgt_dataset = src_tgt_dataset.map(
      lambda src, tgt: (
          src, tgt, tf.size(src), tf.size(tgt)),
      num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

  # Bucket by source sequence length (buckets for lengths 0-9, 10-19, ...)
  def batching_func(x):
    return x.batch(
        batch_size)

  # if num_buckets > 1:
  #
  #   def key_func(unused_1, unused_2, unused_3, src_len, tgt_len):
  #     # Calculate bucket_width by maximum source sequence length.
  #     # Pairs with length [0, bucket_width) go to bucket 0, length
  #     # [bucket_width, 2 * bucket_width) go to bucket 1, etc.  Pairs with length
  #     # over ((num_bucket-1) * bucket_width) words all go into the last bucket.
  #     if src_max_len:
  #       bucket_width = (src_max_len + num_buckets - 1) // num_buckets
  #     else:
  #       bucket_width = 10
  #
  #     # Bucket sentence pairs by the length of their source sentence and target
  #     # sentence.
  #     bucket_id = tf.maximum(src_len // bucket_width, tgt_len // bucket_width)
  #     return tf.to_int64(tf.minimum(num_buckets, bucket_id))
  #
  #   def reduce_func(unused_key, windowed_data):
  #     return batching_func(windowed_data)
  #
  #   batched_dataset = src_tgt_dataset.apply(
  #       tf.contrib.data.group_by_window(
  #           key_func=key_func, reduce_func=reduce_func, window_size=batch_size))
  #
  # else:
  batched_dataset = batching_func(src_tgt_dataset)
  batched_iter = batched_dataset.make_initializable_iterator()
  (src_ids, tgt_ids, src_seq_len,
   tgt_seq_len) = (batched_iter.get_next())
  return BatchedInput(
      initializer=batched_iter.initializer,
      source=src_ids,
      target_input=tgt_ids,
      target_output=tgt_ids,
      source_sequence_length=src_seq_len,
      target_sequence_length=tgt_seq_len)
