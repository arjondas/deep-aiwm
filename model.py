import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from components import Encoder, Decoder, get_Embedder, get_Extractor

class AWM(tf.keras.Model):
  def __init__(self):
    super(AWM, self).__init__()
    self.encoder = Encoder()
    self.reshape1 = layers.Reshape((128, 128, 1))
    self.embedder = get_Embedder()
    self.extractor = get_Extractor()
    self.reshape2 = layers.Reshape((128, 128))
    self.decoder = Decoder()

  def __call__(self, batch_audio, batch_image):
    enc = self.encoder(batch_audio)
    out = self.reshape1(enc)
    marked = self.embedder([out, batch_image])
    out = self.extractor(marked)
    out = self.reshape2(out)
    out = self.decoder(out)
    return out, enc, marked


class AutoEncoder(tf.keras.Model):
  def __init__(self):
    super(AutoEncoder, self).__init__()
    self.encoder = Encoder()
    self.reshape1 = layers.Reshape((128, 128, 1))
    self.reshape2 = layers.Reshape((128, 128))
    self.decoder = Decoder()

  def __call__(self, batch_audio, batch_image):
    enc = self.encoder(batch_audio)
    out = self.reshape1(enc)
    out = self.reshape2(out)
    out = self.decoder(out)
    return out, enc, batch_image
  
  def loss_function(self, pred, orig_audio, marked=None, orig_img=None):
    audio_loss = 100*tf.reduce_mean(tf.keras.losses.mean_square_error(orig_audio, pred))
    return (tf.Variable(0.0), audio_loss)