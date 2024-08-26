# Used this to update outdated codes from the original repo

import glob
filenames = glob.glob('/Users/jacklin/Documents/Oin/openCV/keras_ssd/keras_layers/*.py')
for filename in filenames:
    with open(filename, 'r') as file :
        text = file.read()
    text = text.replace('keras.engine.topology', 'tensorflow.keras.layers')
    text = text.replace('K.image_dim_ordering()', 'K.image_data_format()')
    text = text.replace('._keras_shape', '.shape')
    text = text.replace('.trainable_weights', '._trainable_weights')
    text = text.replace('tf', 'channels_last')
    text = text.replace('keras.backend as K', 'tensorflow as tf')
    text = text.replace('K.', 'tf.keras.backend.')
    with open(filename, 'w') as file:
        file.write(text)
with open('/Users/jacklin/Documents/Oin/openCV/keras_ssd/keras_loss_function/keras_ssd_loss.py', 'r') as file :
    text = file.read()
text = text.replace('from __future__ import division', '')
with open('/Users/jacklin/Documents/Oin/openCV/keras_ssd/keras_loss_function/keras_ssd_loss.py', 'w') as file:
    file.write(text)
with open('/Users/jacklin/Documents/Oin/openCV/keras_ssd/keras_loss_function/keras_ssd_loss.py', 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write('from __future__ import division\nimport tensorflow as tf\ntf.to_float = lambda x: tf.cast(x, tf.float32)'.rstrip('\r\n') + '\n' + content)
with open('/Users/jacklin/Documents/Oin/openCV/keras_ssd/data_generator/object_detection_2d_data_generator.py', 'r') as file :
    text = file.read()
text = text.replace('yield ret', 'yield tuple(ret)')
with open('/Users/jacklin/Documents/Oin/openCV/keras_ssd/data_generator/object_detection_2d_data_generator.py', 'w') as file:
    file.write(text)

with open('/Users/jacklin/Documents/Oin/openCV/keras_ssd/eval_utils/average_precision_evaluator.py', 'r') as file :
    text = file.read()
text = text.replace('if len(predictions) == 0:',
                    'if len(predictions) == 0:\n                cumulative_true_positives.append(1)\n                cumulative_false_positives.append(1)')
with open('/Users/jacklin/Documents/Oin/openCV/keras_ssd/eval_utils/average_precision_evaluator.py', 'w') as file:
    file.write(text)

with open('/Users/jacklin/Documents/Oin/openCV/keras_ssd/keras_loss_function/keras_ssd_loss.py', 'r') as file :
    text = file.read()
text = text.replace('tf.log', 'tf.math.log')
text = text.replace('self.neg_pos_ratio = tf.constant(self.neg_pos_ratio)', '')
text = text.replace('self.n_neg_min = tf.constant(self.n_neg_min)', '')
text = text.replace('self.alpha = tf.constant(self.alpha)', '')
text = text.replace('tf.count_nonzero', 'tf.math.count_nonzero')
text = text.replace('tf.to_int32(n_positive)', 'tf.cast(n_positive, tf.int32)')
with open('/Users/jacklin/Documents/Oin/openCV/keras_ssd/keras_loss_function/keras_ssd_loss.py', 'w') as file:
    file.write(text)

import os
os.environ['PYTHONPATH'] += ':/Users/jacklin/Documents/Oin/openCV/keras_ssd/:/Users/jacklin/Documents/Oin/openCV/keras_ssd/keras_layers/'
