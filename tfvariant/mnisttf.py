import tensorflow as tf
import tensorflow_datasets as tfds

# If a TensorFlow operation has both CPU and GPU implementations, by default, the GPU device is prioritized 
# when the operation is assigned. 

def main():


    # Uncomment the first 2 lines to see if tf uses the GPU and what GPU it uses 

    #print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    #print(tf.config.list_physical_devices('GPU'))
    (train, test), ds_info = tfds.load(
            'mnist',
            split = ['train', 'test'],
            shuffle_files = True,  # shuffles the input files
            as_supervised = True, # if load returns 2-tuple with structure (input, label)
            with_info = True, # information about the dataset
    
    )

    train = train.map(normalize_data, num_parallel_calls=tf.data.AUTOTUNE) # num_parallel_calls just does things in parallel

    train = train.cache() # cache the data for better performance
    train = train.shuffle(ds_info.splits['train'].num_examples) # true randomness when shuffling
    train = train.batch(128) # divide the data into 128 images chucks 
    # perform backprop on every chuck
    train = train.prefetch(tf.data.AUTOTUNE) # something for performance

    # same for test data (maybe should do a function)

    test = test.map(normalize_data, num_parallel_calls=tf.data.AUTOTUNE)
    test = test.batch(128)
    test = test.cache()
    test = test.prefetch(tf.data.AUTOTUNE)


    model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape =(28, 28)), # basically exactly what i am doing as 28 * 28 = 784
            tf.keras.layers.Dense(100, activation = 'relu'), # i am using the sigmoid function as activation, not relu 
            tf.keras.layers.Dense(10), # basic output (10 digits, 10 output neurons)
        ])


    model.compile(
            optimizer = tf.keras.optimizers.Adam(0.001), # no idea what Adam is, but I assume it has to do with the learning 
            # rate
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True), # ... 
            # again, not sure about Spare Categorical but I am indeed using in the optimized version the Cross Entropy 
            # function as the cost function f(x) = -(ylog(x) + (1 - y)log(x)) (and the sum over this)
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
        )

    model.fit(
            train, 
            epochs = 100,
            validation_data = test
        )



def normalize_data(image, label):
    """
    uint8 -> float32

    Normalize the data to be easier to train on. New values in [0, 1]
    """

    return tf.cast(image, tf.float32) / 255.0, label 



if __name__ == '__main__':
    main()


