'''
This file contains models based on "Merge" architecture

Avaliable hyperparameters are
- Sentence length
- Vocabulary size
- Feature vector length (Image)
- Embedded word vector length 
- Merge function (one of 'Add', 'Mult', 'Mixed')
- Flavor of RNN (one of 'LSTM' or 'GRU')

'''
from tensorflow.keras import Model
from tensorflow.keras.layers import Input,Dense,LSTM,Embedding,Dropout,Add,Multiply, GRU

def make(seq_len, vocab_size, feat_len, embed_vec_len, merge_function, rnn_type):
    '''
    Build a Merge model
    seq_len - (int) maximum length of a sentence
    vocab_size - (int) vocabulary size, needed for output dimention
    feat_len - (int) expected size of a image feature vector
    embed_vec_len - (int) size of word embedding. Bigger values yeld better results at a cost of computation time
    merge_function - (string) How to merge feature vector and rnn output. (one of 'Add', 'Mult', 'Mixed')
    rnn_type - (string) Flavor of RNN (one of 'LSTM' or 'GRU')

    Model accepts outputs of shape [(b, feat_len), (b, seq_len)]
    '''

    inputs1 = Input(shape=(feat_len,), name='fe_input')
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(embed_vec_len, activation='relu')(fe1)

    inputs2 = Input(shape=(seq_len,), name='seq_input')
    se1 = Embedding(input_dim=vocab_size+1,
                    output_dim=embed_vec_len,
                    input_length=seq_len,
                    mask_zero=True,
                    name='embed_input')(inputs2)
    se2 = Dropout(0.5)(se1)

    if rnn_type == 'GRU':
        se3 = GRU(embed_vec_len, return_sequences=ATT)(se2)
    elif rnn_type == 'LSTM':
        se3 = LSTM(embed_vec_len, return_sequences=ATT)(se2)
    else:
        print(f"Wrong type of RNN given: {rnn_type}. Expected 'GRU' or 'LSTM' ")
        exit(-1)
    
    if merge_function == "Add":
        decoder1 = Add()([fe2, se3])
    elif merge_function == "Mult":
        decoder1 = Multiply()([fe3, se3])
    elif merge_function == "Mixed":
        att1 = Dense(embed_vec_len)(se3)
        fe3 = Multiply()([att1,fe2])
        decoder1 = Add()([fe3, se3])
     else:
        print(f"Wrong type of Merge function given: {merge_function}. Expected 'Add', 'Mult' or 'Mixed' ")
        exit(-1)
    
    decoder2 = Dense(embed_vec_len*2, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
    return model

