import tensorflow as tf
import math

class FirstSirenLayer(tf.keras.Model):
    def __init__(self, in_f, out_f, frequency_factor=30):
        super(FirstSirenLayer, self).__init__()
        self.in_f = in_f
        self.out_f = out_f
        self.frequency_factor = frequency_factor
        
        u = 1/self.in_f
        self.dense = tf.keras.layers.Dense(out_f, input_shape=(in_f,), kernel_initializer=tf.keras.initializers.RandomUniform(minval=-u, maxval=u))

    def call(self, x):
        x = self.dense(x)
        return tf.sin(self.frequency_factor * x)
        
        
class MiddleSirenLayer(tf.keras.Model):
    def __init__(self, in_f, out_f, frequency_factor=30):
        super(MiddleSirenLayer, self).__init__()
        self.in_f = in_f
        self.out_f = out_f
        self.frequency_factor = frequency_factor

        u = tf.sqrt(6 / self.in_f) / self.frequency_factor

        self.dense = tf.keras.layers.Dense(out_f, input_shape=(in_f,),kernel_initializer=tf.keras.initializers.RandomUniform(minval=-u, maxval=u))

    def call(self, x):
        x = self.dense(x)
        return tf.sin(self.frequency_factor * x)
        
        
class FinalSirenLayer(tf.keras.Model):
    def __init__(self, in_f, out_f, frequency_factor=30, use_bias=False):
        super(FinalSirenLayer, self).__init__()
        self.in_f = in_f
        self.out_f = out_f
        self.frequency_factor = frequency_factor
        self.use_bias = use_bias

        u = math.sqrt(6 / self.in_f) / self.frequency_factor
        
        self.dense = tf.keras.layers.Dense(out_f, input_shape=(in_f,),kernel_initializer=tf.keras.initializers.RandomUniform(minval=-u, maxval=u), use_bias=self.use_bias)

    def call(self, x):
        x = self.dense(x)
        return x
        
class Exponential_BSNN(tf.keras.Model):
    def __init__(self, in_f, out_f, is_last=False, use_bias=False):
        super(Exponential_BSNN, self).__init__()
        self.in_f = in_f
        self.out_f = out_f
        self.is_last = is_last
        self.use_bias = use_bias
        
        self.dense = tf.keras.layers.Dense(out_f, input_shape=(in_f,), kernel_initializer=tf.keras.initializers.Orthogonal(gain=1.0), use_bias=self.use_bias)

    def call(self, x):
        x = self.dense(x)
        return x if self.is_last else tf.exp(x-1.0)
        
class Sinusoidal_BSNN(tf.keras.Model):
    def __init__(self, in_f, out_f, is_last=False, use_bias=False):
        super(Sinusoidal_BSNN, self).__init__()
        self.in_f = in_f
        self.out_f = out_f
        self.is_last = is_last
        self.use_bias = use_bias
        
        self.dense = tf.keras.layers.Dense(out_f, input_shape=(in_f,), kernel_initializer=tf.keras.initializers.Orthogonal(gain=1.0), use_bias=self.use_bias)

    def call(self, x):
        x = self.dense(x)
        return x if self.is_last else tf.sqrt(2.0)*tf.sin(math.pi/4.0 + x)
        
