# CNN-Text Classification
 Turkish Text Classification using Convolutional Neural Networks with similar architectures described by Yoon Kim
 
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
 
