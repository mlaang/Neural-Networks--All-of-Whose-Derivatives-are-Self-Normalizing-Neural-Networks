import tensorflow as tf
import math

#Implements the positional encoder of Mildenhall et al.
class PositionalEncoderLayer(tf.keras.Model):
    def __init__(self, in_f, out_f, scale_factor = 1.0):
        super(PositionalEncoderLayer, self).__init__()
        self.in_f = in_f
        self.out_f = out_f
        self.scale_factor = scale_factor
        
        if in_f != 2:
            print("in_f must be 2.")
        if out_f % 4 != 0:
            print("out_f must be divisible by 4.")
            
        self.useful_constant = tf.constant(tf.pow(2.0, tf.range(out_f/4, dtype=tf.float32)))
        
    def call(self, x):
        u = tf.einsum("i,jk->jki", self.useful_constant, x)
        
        a = tf.sqrt(2.0*self.scale_factor)*tf.sin(u)
        b = tf.sqrt(2.0*self.scale_factor)*tf.cos(u)
        
        c = tf.stack([a,b],axis=1)
        return tf.reshape(c, [-1, self.out_f])
        
#Modified positional encoder that applies the positional encoder of Mildenhall et al. to three different co-ordinates constructed by rotation
class RotatedPositionalEncoderLayer(tf.keras.Model):
    def __init__(self, in_f, out_f, scale_factor = 1.0):
        super(RotatedPositionalEncoderLayer, self).__init__()
        self.in_f = in_f
        self.out_f = out_f
        self.scale_factor = scale_factor
        
        if in_f != 2:
            raise Exception("Error in RotatedPositionalEncoder: in_f must be 2, is " + str(in_f) + ".")
        if out_f % 12 != 0:
            raise Exception("Error in RotatedPositionalEncoder: out_f must be divisible by 12, is " + str(out_f) + ".")
            
        self.useful_constant = tf.constant(tf.pow(2.0, tf.range(out_f/12, dtype=tf.float32)))
        self.rotation_matrix_1 = tf.constant(tf.reshape([[tf.cos(2.0*math.pi/3.0), tf.sin(2.0*math.pi/3.0)], [tf.sin(2.0*math.pi/3.0), tf.cos(2*math.pi/3.0)]], [2, 2]),dtype=tf.float32)
        self.rotation_matrix_2 = tf.constant(tf.reshape(self.rotation_matrix_1 * self.rotation_matrix_1, [2,2]),dtype=tf.float32)

    def call(self, x):
        a = x
        b = tf.matmul(x, self.rotation_matrix_1)
        c = tf.matmul(x, self.rotation_matrix_2)
        
        ua = tf.einsum("i,jk->jki", self.useful_constant, a)
        ub = tf.einsum("i,jk->jki", self.useful_constant, b)
        uc = tf.einsum("i,jk->jki", self.useful_constant, c)
        
        uaa = tf.sqrt(2.0*self.scale_factor)*tf.cos(ua)
        uab = tf.sqrt(2.0*self.scale_factor)*tf.sin(ua)
        
        uba = tf.sqrt(2.0*self.scale_factor)*tf.cos(ub)
        ubb = tf.sqrt(2.0*self.scale_factor)*tf.sin(ub)
        
        uca = tf.sqrt(2.0*self.scale_factor)*tf.cos(uc)
        ucb = tf.sqrt(2.0*self.scale_factor)*tf.sin(uc)
 
        stack = tf.stack([uaa, uab, uba, ubb, uca, ucb], axis=1)
        return tf.reshape(stack, [-1, self.out_f])
        