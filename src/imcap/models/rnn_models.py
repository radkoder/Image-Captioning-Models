'''
This file contains models based on LSTM layer

Avaliable hyperparameters are
- Sentence length
- Vocabulary size
- Feature vector length (Image)
- Embedded word vector length 
- Flavor of RNN (one of 'LSTM' or 'GRU')

'''
from tensorflow.keras import Model
from tensorflow.keras.layers import Input,Dense,LSTM,Embedding,GRU

def make(seq_len,vocab_size, feat_len, embed_vec_len, rnn_type):
    '''
    Build a LSTM model
    seq_len - (int) maximum length of a sentence
    vocab_size - (int) vocabulary size, needed for output dimention
    feat_len - (int) expected size of a image feature vector
    embed_vec_len - (int) size of word embedding. Bigger values yeld better results at a cost of computation time
    rnn_type - (string) Flavor of RNN (one of 'LSTM' or 'GRU')

    Model accepts outputs of shape [(b, feat_len), (b, seq_len)]
    '''
    inputs1 = Input(shape=(feat_len,), name='fe_input')   
    is1 = Dense(embed_vec_len)(inputs1)
    is2 = Dense(embed_vec_len)(inputs1)

    inputs2 = Input(shape=(seq_len,), name='seq_input')
    se1 = Embedding(input_dim=vocab_size+1,
                    output_dim=embed_vec_len,
                    input_length=seq_len,
                    mask_zero=True,
                    name='embed_input')(inputs2)

    if rnn_type == 'GRU':
        se3 = GRU(embed_vec_len, dropout =0.5)(se1, initial_state = [is1,is2])
    elif rnn_type == 'LSTM':
        se3 = LSTM(embed_vec_len, dropout =0.5)(se1, initial_state = [is1,is2])
    else:
        print(f"Wrong type of RNN given: {rnn_type}. Expected 'GRU' or 'LSTM' ")
        exit(-1)
    outputs = Dense(vocab_size, activation='softmax')(se3)
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
    return model
    
