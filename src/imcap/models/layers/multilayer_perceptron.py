def MLP(hidden_dim, embed_dim, rate=0.2):
    model=tf.keras.Sequential(
            [   tf.keras.layers.Dense(hidden_dim, activation=gelu),
                tf.keras.layers.Dropout(rate),
                tf.keras.layers.Dense(embed_dim, activation=gelu)
            ]
        )
    return model