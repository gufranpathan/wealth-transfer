import tensorflow as tf

# Declare parameters that you want to iterate over as a list of dict

model_hyperparams= [
    {
        'fine_tune' :16,
        'reg_amount':1e-5,
        'n_epochs':1000,
        'batch_size':32,
        'AUTOTUNE':tf.data.experimental.AUTOTUNE,
        'patience':5,
        'RGB_only': True
    } # can add more iterations of params separated by a comma in the list



]