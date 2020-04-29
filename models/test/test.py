import tensorflow as tf

from databases import MiniImagenetDatabase


model = tf.keras.applications.VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))

flatten = tf.keras.layers.Flatten(name='flatten')(model.output)
fc1 = tf.keras.layers.Dense(512)(flatten)
fc2 = tf.keras.layers.Dense(512)(fc1)
fc3 = tf.keras.layers.Dense(5)(fc2)

new_model = tf.keras.models.Model(inputs=[model.input], outputs=[fc3])
new_model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy())


data_base = MiniImagenetDatabase(input_shape=(224, 224, 3))

dataset = data_base.get_supervised_meta_learning_dataset(data_base.train_folders, 5, 1, 1, 2)

for item in dataset:
    x, y = item
    x1, x2 = x
    y1, y2 = y
    x1 = tf.reshape(x1, (10, 224, 224, 3))
    y1 = tf.reshape(y1, (10, 5))
    new_model.fit(x1, y1)
    print(tf.argmax(new_model.predict(x1), axis=0))
    print(tf.argmax(y1, axis=0))
    break
