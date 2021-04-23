import miniml

# create model
model = miniml.Model()
model.conv2d(8, 5, 1, activation=None)
model.maxpool(2, 2)
model.activation('relu')
model.flatten()
model.dense(32, 'relu', 'he')
model.dropout(0.7)
model.dense(1, 'sigmoid', 'plain')

# print summary
model.summary((64, 64, 3))
