import Layers
import PositionalEncoders
import tensorflow as tf

def build_SIREN_model(dimensions):
    actual_model = tf.keras.Sequential()
    actual_model.add(Layers.FirstSirenLayer(dimensions[0], dimensions[1]))
    other_layers = []
    for dim0, dim1 in zip(dimensions[1:-2], dimensions[2:-1]):
        actual_model.add(Layers.MiddleSirenLayer(dim0, dim1))
    actual_model.add(Layers.FinalSirenLayer(dimensions[-2], dimensions[-1]))
    return actual_model
    
def build_orthogonal_model_with_SIREN_encoder(dimensions, use_bias=False):
    actual_model = tf.keras.Sequential()
    actual_model.add(Layers.FirstSirenLayer(dimensions[0], dimensions[1]))
    other_layers = []
    for dim0, dim1 in zip(dimensions[1:-2], dimensions[2:-1]):
        actual_model.add(Layers.Sinusoidal_BSNN(dim0, dim1, use_bias=use_bias))
    actual_model.add(Layers.Sinusoidal_BSNN(dimensions[-2], dimensions[-1], is_last=True, use_bias=use_bias))
    return actual_model

# dimension[0] must be 2 (due to implementation)
# dimension[1] must be divisible by 4
def build_orthogonal_model_with_Mildenhall_et_al_encoder(dimensions, use_bias=False, scale_factor=1.0):
    actual_model = tf.keras.Sequential()
    actual_model.add(PositionalEncoders.PositionalEncoderLayer(dimensions[0], dimensions[1], scale_factor=scale_factor))
    other_layers = []
    for dim0, dim1 in zip(dimensions[1:-2], dimensions[2:-1]):
        actual_model.add(Layers.Sinusoidal_BSNN(dim0, dim1))
    actual_model.add(Layers.Sinusoidal_BSNN(dimensions[-2], dimensions[-1], is_last=True))
    return actual_model
    
# dimension[0] must be 2
# dimension[1] must be divisible by 12
def build_orthogonal_model_with_rotated_encoder(dimensions, use_bias=False, scale_factor=1.0):
    actual_model = tf.keras.Sequential()
    actual_model.add(PositionalEncoders.RotatedPositionalEncoderLayer(dimensions[0], dimensions[1], scale_factor=scale_factor))
    other_layers = []
    for dim0, dim1 in zip(dimensions[1:-2], dimensions[2:-1]):
        actual_model.add(Layers.Sinusoidal_BSNN(dim0, dim1))
    actual_model.add(Layers.Sinusoidal_BSNN(dimensions[-2], dimensions[-1], is_last=True))
    return actual_model