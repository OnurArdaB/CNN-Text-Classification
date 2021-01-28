# CNN-Text Classification
 Turkish Text Classification using Convolutional Neural Networks with similar architectures described by Yoon Kim.
 
# Architectures 
 ## CNN-rand
 ```python
 def rand_model():

    inputs1 = Input(shape=(450, ))
    embedding1 = Embedding(input_dim=7000,
                           output_dim=32,
                           trainable=True, input_length=450)(inputs1)
    conv1 = Conv1D(filters=16, kernel_size=3, activation='relu'
                   )(embedding1)
    pool1 = MaxPooling1D()(conv1)
    flat1 = Flatten()(pool1)
    
    conv2 = Conv1D(filters=16, kernel_size=4, activation='relu'
                   )(embedding1)
    pool2 = MaxPooling1D()(conv2)
    flat2 = Flatten()(pool2)

    
    conv3 = Conv1D(filters=16, kernel_size=5, activation='relu'
                   )(embedding1)
    pool3 = MaxPooling1D()(conv3)
    flat3 = Flatten()(pool3)
    
    # merge

    merged = concatenate([flat1,flat2,flat3])

    # interpretation
    drop1 = Dropout(0.5)(merged)
    dense1 = Dense(64, activation='relu')(drop1)
    outputs = Dense(5, activation='softmax')(dense1)
    model = Model(inputs=[inputs1], outputs=outputs)

    # compile

    model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])

    # summarize

    print(model.summary())
    return model
 
 ```
 
This model has an embedding layer that contains randomly generated embeddings trained with the entire network. It can be observed that ``` trainable=True``` allows the network to train the embedding layer which is set to False in cases where an pre-trained embedding layer exists. LSTM layer for *interpretation* part also showed a performance increase.

## CNN-static
 ```python
 def static_model():

    inputs1 = Input(shape=(450, ))
    embedding1 = Embedding(input_dim=pretrained_weight_2.shape[0],
                           output_dim=400,
                           weights=[pretrained_weight_2],
                           trainable=False, input_length=450)(inputs1)
    conv1 = Conv1D(filters=16, kernel_size=2, activation='relu'
                   )(embedding1)
    pool1 = GlobalMaxPooling1D()(conv1)

    flat1 = Flatten()(pool1)

    
    conv2 = Conv1D(filters=16, kernel_size=3, activation='relu'
                   )(embedding1)
    pool2 = GlobalMaxPooling1D()(conv2)
    flat2 = Flatten()(pool2)

    
    conv3 = Conv1D(filters=16, kernel_size=4, activation='relu'
                   )(embedding1)
    pool3 = GlobalMaxPooling1D()(conv3)

    flat3 = Flatten()(pool3)

    # merge

    merged = concatenate([flat1,flat2,flat3])

    # interpretation
    dense = Dense(32, activation='relu')(merged)
    drop1 = Dropout(0.2)(dense)
    outputs = Dense(5, activation='softmax')(drop1)
    model = Model(inputs=[inputs1], outputs=outputs)

    # compile

    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])

    # summarize

    print(model.summary())
    return model
 
 ```

Embedding layer is kept static and MaxPooling1D is changed with GlobalMaxPooling1D. Architectures can be modified with respect to different problems but the main structure is as below. This model contains 3 separate Convolutional Layers combined with Pooling Layers. Results are later concatenated and fed to a sequence of Fully Connected Layers.

## CNN-non-static
```python
def non_static_model():

    inputs1 = Input(shape=(450, ))
    embedding1 = Embedding(input_dim=pretrained_weight_2.shape[0],
                           output_dim=400,
                           weights=[pretrained_weight_2],
                           trainable=True, input_length=450)(inputs1)
    conv1 = Conv1D(filters=32, kernel_size=2, activation='relu'
                   )(embedding1)
    pool1 = GlobalMaxPooling1D()(conv1)
    flat1 = Flatten()(pool1)

    
    conv2 = Conv1D(filters=32, kernel_size=3, activation='relu'
                   )(embedding1)
    pool2 = GlobalMaxPooling1D()(conv2)
    flat2 = Flatten()(pool2)

    
    conv3 = Conv1D(filters=32, kernel_size=4, activation='relu'
                   )(embedding1)
    pool3 = GlobalMaxPooling1D()(conv3)
    flat3 = Flatten()(pool3)

    # merge

    merged = concatenate([flat1,flat2,flat3])

    # interpretation
    dense = Dense(32, activation='relu')(merged)
    drop1 = Dropout(0.2)(dense)
    outputs = Dense(5, activation='softmax')(drop1)
    model = Model(inputs=[inputs1], outputs=outputs)

    # compile

    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])

    # summarize

    print(model.summary())
    return model
```
 Similar to static model with only embedding layer is trained with the entire network by setting ``` trainable=True```.
 
