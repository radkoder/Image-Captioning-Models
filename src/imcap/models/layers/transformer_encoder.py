import multihead_attention
import tensorflow as tf

class TransformerEncoderBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, mlp_hidden_dim):
        super(TransformerEncoderBlock, self).__init__()
        self.mlp = MLP(mlp_hidden_dim, embed_dim,0.2)
        self.MHA_layer = MultiHeadSelfAttention(embed_dim, num_heads)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(0.2)
        self.dropout2 = tf.keras.layers.Dropout(0.2)

    def call(self, input_embeddings, training=True):
        input_embeddings_norm = self.layernorm1(input_embeddings)
        output = self.MHA_layer(input_embeddings_norm)
        output = self.dropout1(output, training=training)
        output_1 = output + input_embeddings
        #Skip Connection: Adding input_embeddings to the output 

        output_norm = self.layernorm2(output_1)
        MLP_output = self.mlp(output_norm)
        MLP_output = self.dropout2(MLP_output, training=training)
        return MLP_output + output_1 
        #Skip Connection: Adding output_1 to the final output MLP_output 