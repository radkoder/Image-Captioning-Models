def gelu(x):
    """ The GELU Activation function: defined as x*CDF(x) for the Standard Normal(0,1) Distribution"""

    return 0.5 * x * (1.0 + tf.math.erf(x / tf.cast(tf.sqrt(2.0), x.dtype)))