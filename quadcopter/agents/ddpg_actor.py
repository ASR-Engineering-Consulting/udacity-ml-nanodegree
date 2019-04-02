from keras import layers, models, optimizers, regularizers
from keras import backend as K
from keras.utils.vis_utils import plot_model


class Actor:
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, action_low, action_high):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            action_low (array): Min value of each action dimension
            action_high (array): Max value of each action dimension
        """
        self.state_size = state_size
        self.action_size = action_size
        self.action_low = action_low
        self.action_high = action_high
        self.action_range = self.action_high - self.action_low

        # Initialize any other variables here
        self.lr = 0.0001
        self.l2 = 1e-6
        self.units_block_1 = 400
        self.units_block_2 = 300
        self.units_block_3 = 200
    
        self.build_model()

    def build_model(self):
        """Build an actor (policy) network that maps states -> actions."""
        # Define input layer (states)
        states = layers.Input(shape=(self.state_size,), name='states')

        # Add hidden layers
        
        # first block
        # increase units and regularizers according to research paper
        net = layers.Dense(units=self.units_block_1, kernel_regularizer=regularizers.l2(self.l2))(states)
        # add batchnorm with default parameters
        net = layers.BatchNormalization()(net)
        # add relu activation
        net = layers.Activation('relu')(net)
                           
        # second block
        net = layers.Dense(units=self.units_block_2, kernel_regularizer=regularizers.l2(self.l2))(net)
        net = layers.BatchNormalization()(net)
        net = layers.Activation('relu')(net)
        
        # third block
        net = layers.Dense(units=self.units_block_3, kernel_regularizer=regularizers.l2(self.l2))(net)
        net = layers.BatchNormalization()(net)
        net = layers.Activation('relu')(net)

        # Add final output layer with sigmoid activation
        raw_actions = layers.Dense(units=self.action_size, activation='sigmoid',
            name='raw_actions')(net)

        # Scale [0, 1] output for each action dimension to proper range
        actions = layers.Lambda(lambda x: (x * self.action_range) + self.action_low,
            name='actions')(raw_actions)

        # Create Keras model
        self.model = models.Model(inputs=states, outputs=actions)

        # Define loss function using action value (Q value) gradients
        action_gradients = layers.Input(shape=(self.action_size,))
        loss = K.mean(-action_gradients * actions)

        # Incorporate any additional losses here (e.g. from regularizers)

        # Define optimizer and training function
        # set learning rate to 10**-4 (0.0001) as recommend by research paper
        optimizer = optimizers.Adam(lr=self.lr)
        updates_op = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
        self.train_fn = K.function(
            inputs=[self.model.input, action_gradients, K.learning_phase()],
            outputs=[],
            updates=updates_op)
