import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell, MultiRNNCell, OutputProjectionWrapper, ResidualWrapper
from tensorflow.contrib.seq2seq import BasicDecoder, BahdanauAttention, AttentionWrapper
from text.symbols import symbols
from util.infolog import log
from .helpers import TacoTestHelper, TacoTrainingHelper
from .modules import encoder_cbhg, post_cbhg, prenet
from .rnn_wrappers import DecoderPrenetWrapper, ConcatOutputAndAttentionWrapper



class Tacotron():
  def __init__(self):


  def initialize(self, inputs, input_lengths, mel_targets=None, linear_targets=None):
    with tf.variable_scope('inference') as scope:
      is_training = linear_targets is not None
      batch_size = tf.shape(inputs)[0]
      # Embeddings
      embedding_table = tf.get_variable(
        'embedding', [len(symbols), 256], dtype=tf.float32,
        initializer=tf.truncated_normal_initializer(stddev=0.5))
      embedded_inputs = tf.nn.embedding_lookup(embedding_table, inputs)          # [N, T_in, embed_depth=256]
      # Encoder
      prenet_outputs = prenet(embedded_inputs, is_training, [256, 128])    # [N, T_in, prenet_depths[-1]=128]
      encoder_outputs = encoder_cbhg(prenet_outputs, input_lengths, is_training, # [N, T_in, encoder_depth=256]
                                    256)
      # Attention
      attention_cell = AttentionWrapper(
        GRUCell(256),
        BahdanauAttention(256, encoder_outputs),
        alignment_history=True,
        output_attention=False)                                                  # [N, T_in, attention_depth=256]
      
      # Apply prenet before concatenation in AttentionWrapper.
      attention_cell = DecoderPrenetWrapper(attention_cell, is_training, [256, 128])
      # Concatenate attention context vector and RNN cell output into a 2*attention_depth=512D vector.
      concat_cell = ConcatOutputAndAttentionWrapper(attention_cell)              # [N, T_in, 2*attention_depth=512]
      # Decoder (layers specified bottom to top):
      decoder_cell = MultiRNNCell([
          OutputProjectionWrapper(concat_cell, 256),
          ResidualWrapper(GRUCell(256)),
          ResidualWrapper(GRUCell256))
        ], state_is_tuple=True)                                                  # [N, T_in, decoder_depth=256]
      # Project onto r mel spectrograms (predict r outputs at each RNN step):
      output_cell = OutputProjectionWrapper(decoder_cell,80 *5)
      decoder_init_state = output_cell.zero_state(batch_size=batch_size, dtype=tf.float32)

      if is_training:
        helper = TacoTrainingHelper(inputs, mel_targets, 80,5)
      else:
        helper = TacoTestHelper(batch_size, 80, 5)

      (decoder_outputs, _), final_decoder_state, _ = tf.contrib.seq2seq.dynamic_decode(
        BasicDecoder(output_cell, helper, decoder_init_state),
        maximum_iterations=200)                                         # [N, T_out/r, M*r]

      # Reshape outputs to be one output per entry
      mel_outputs = tf.reshape(decoder_outputs, [batch_size, -1, 80])   # [N, T_out, M]

      # Add post-processing CBHG:
      post_outputs = post_cbhg(mel_outputs, 80, is_training,            # [N, T_out, postnet_depth=256]
                               256)
      linear_outputs = tf.layers.dense(post_outputs, 1025)                # [N, T_out, F]

      # Grab alignments from the final decoder state:
      alignments = tf.transpose(final_decoder_state[0].alignment_history.stack(), [1, 2, 0])

      self.inputs = inputs
      self.input_lengths = input_lengths
      self.mel_outputs = mel_outputs
      self.linear_outputs = linear_outputs
      self.alignments = alignments
      self.mel_targets = mel_targets
      self.linear_targets = linear_targets
      log('Initialized Tacotron model. Dimensions: ')
      log('  embedding:               %d' % embedded_inputs.shape[-1])
      log('  prenet out:              %d' % prenet_outputs.shape[-1])
      log('  encoder out:             %d' % encoder_outputs.shape[-1])
      log('  attention out:           %d' % attention_cell.output_size)
      log('  concat attn & out:       %d' % concat_cell.output_size)
      log('  decoder cell out:        %d' % decoder_cell.output_size)
      log('  decoder out (%d frames):  %d' % (5, decoder_outputs.shape[-1]))
      log('  decoder out (1 frame):   %d' % mel_outputs.shape[-1])
      log('  postnet out:             %d' % post_outputs.shape[-1])
      log('  linear out:              %d' % linear_outputs.shape[-1])


  def add_loss(self):
    '''Adds loss to the model. Sets "loss" field. initialize must have been called.'''
    with tf.variable_scope('loss') as scope:
      self.mel_loss = tf.reduce_mean(tf.abs(self.mel_targets - self.mel_outputs))
      l1 = tf.abs(self.linear_targets - self.linear_outputs)
      # Prioritize loss for frequencies under 3000 Hz.
      n_priority_freq = int(3000 / (20000 * 0.5) * 1025)
      self.linear_loss = 0.5 * tf.reduce_mean(l1) + 0.5 * tf.reduce_mean(l1[:,:,0:n_priority_freq])
      self.loss = self.mel_loss + self.linear_loss


  def add_optimizer(self, global_step):
    '''Adds optimizer. Sets "gradients" and "optimize" fields. add_loss must have been called.

    Args:
      global_step: int32 scalar Tensor representing current global step in training
    '''
    with tf.variable_scope('optimizer') as scope:
      if True:
        self.learning_rate = _learning_rate_decay(0.002, global_step)
      else:
        self.learning_rate = tf.convert_to_tensor(0.002)
      optimizer = tf.train.AdamOptimizer(self.learning_rate, 0.9, 0.999)
      gradients, variables = zip(*optimizer.compute_gradients(self.loss))
      self.gradients = gradients
      clipped_gradients, _ = tf.clip_by_global_norm(gradients, 1.0)

      # Add dependency on UPDATE_OPS; otherwise batchnorm won't work correctly. See:
      # https://github.com/tensorflow/tensorflow/issues/1122
      with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        self.optimize = optimizer.apply_gradients(zip(clipped_gradients, variables),
          global_step=global_step)


def _learning_rate_decay(init_lr, global_step):
  # Noam scheme from tensor2tensor:
  warmup_steps = 4000.0
  step = tf.cast(global_step + 1, dtype=tf.float32)
  return init_lr * warmup_steps**0.5 * tf.minimum(step * warmup_steps**-1.5, step**-0.5)
