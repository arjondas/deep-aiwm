import tensorflow as tf
from tensorflow.keras import layers, Model, Input

class Encoder(tf.keras.Model):
  def __init__(self):

    """
      Expects input shape of (None, 8192, 1)
    """

    super(Encoder, self).__init__()

    self.audio_encoder_model = tf.keras.Sequential(name='Encoder')

    self.audio_encoder_model.add(layers.Reshape((128, 64)))
    self.audio_encoder_model.add(layers.LSTM(128, return_sequences=True))
    self.audio_encoder_model.add(layers.LSTM(128, activation='sigmoid', return_sequences=True))

  def call(self, input_audio):
    out = self.audio_encoder_model(input_audio)
    return out

class Decoder(tf.keras.Model):
  def __init__(self):

    """
      Expects feature shape of (None, 128, 128)
    """

    super(Decoder, self).__init__()

    self.audio_decoder_model = tf.keras.Sequential(name='Decoder')

    self.audio_decoder_model.add(layers.LSTM(64, return_sequences=True, input_shape=(128, 128)))
    self.audio_decoder_model.add(layers.LSTM(64, return_sequences=True, activation='sigmoid'))
    self.audio_decoder_model.add(layers.Reshape((8192, 1)))

  def call(self, encoded_feature_input):
    out = self.audio_decoder_model(encoded_feature_input)
    return out


def get_Embedder():
  xa = Input((128, 128, 1))
  xi = Input((128, 128, 3))
  
  ## Audio
  y = layers.Conv2D(8, 3, padding='same', activation='relu')(xa)
  y = layers.Conv2D(16, 3, padding='same', activation='relu')(y)
  y = layers.Conv2D(32, 3, padding='same', activation='relu')(y)
  y = layers.Conv2D(64, 3, padding='same', activation='relu')(y)

  ## Image
  y_ = layers.Conv2D(8, 3, padding='same', activation='relu')(xi)
  y_ = layers.Conv2D(16, 3, padding='same', activation='relu')(y_)
  y_ = layers.Conv2D(32, 3, padding='same', activation='relu')(y_)
  y_ = layers.Conv2D(64, 3, padding='same', activation='relu')(y_)
  
  y = layers.concatenate([y_, y])
  y = layers.Dense(512, activation='relu')(y)

  y = layers.Conv2D(128, 3, padding='same', activation='relu')(y)
  y = layers.Conv2D(64, 3, padding='same', activation='relu')(y)
  y = layers.Conv2D(32, 3, padding='same', activation='relu')(y)
  y = layers.Conv2D(16, 3, padding='same', activation='relu')(y)
  y = layers.Conv2D(8, 3, padding='same', activation='relu')(y)
  y = layers.Conv2D(4, 3, padding='same', activation='relu')(y)
  y = layers.Conv2D(3, 1, padding='same', activation='sigmoid')(y)

  model = Model(inputs=[xa, xi], outputs=y, name='Embedder')
  return model


def get_Extractor():
  x = Input((128, 128, 3))
  y = layers.Conv2D(8, 3, padding='same', activation='relu')(x)
  y1 = layers.Conv2D(16, 3, padding='same', activation='relu')(y)
  y = layers.Conv2D(32, 3, padding='same', activation='relu')(y1)
  y2 = layers.Conv2D(64, 3, padding='same', activation='relu')(y)
  y = layers.Conv2D(128, 3, padding='same', activation='relu')(y2)
  
  y = layers.Dense(512, activation='relu')(y)
  y = layers.Conv2D(128, 3, padding='same', activation='relu')(y)

  y = layers.Conv2D(64, 3, padding='same')(y)
  y = layers.Add()([y2, y])
  y = layers.Activation('relu')(y)
  y = layers.BatchNormalization()(y)

  y = layers.Conv2D(32, 3, padding='same', activation='relu')(y)

  y = layers.Conv2D(16, 3, padding='same')(y)
  y = layers.Add()([y1, y])
  y = layers.Activation('relu')(y)
  y = layers.BatchNormalization()(y)

  y = layers.Conv2D(8, 3, padding='same', activation='relu')(y)
  y = layers.Conv2D(4, 3, padding='same', activation='relu')(y)
  y = layers.Conv2D(1, 1, padding='same', activation='sigmoid')(y)

  model = Model(inputs=x, outputs=y, name='Extractor')
  return model