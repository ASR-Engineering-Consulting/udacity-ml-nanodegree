from keras import layers, models, optimizers, regularizers
from keras import backend as K


class Critic:
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
        """
        self.state_size = state_size
        self.action_size = action_size

        # Initialize any other variables here
        self.lr = 0.001
        self.l2 = 1e-6
        self.units_block_1 = 400
        self.units_block_2 = 300
        self.units_block_3 = 200

        self.build_model()

    def build_model(self):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        # Define input layers
        states = layers.Input(shape=(self.state_size,), name='states')
        actions = layers.Input(shape=(self.action_size,), name='actions')

        # Add hidden layer(s) for state pathway
        # apply pattern from ddpg_actor
        
        # first block
        net_states = layers.Dense(units=self.units_block_1, kernel_regularizer=regularizers.l2(self.l2))(states)
        net_states = layers.BatchNormalization()(net_states)
        net_states = layers.Activation('relu')(net_states)
        
        # second block
        net_states = layers.Dense(units=self.units_block_2, kernel_regularizer=regularizers.l2(self.l2))(net_states)
        net_states = layers.BatchNormalization()(net_states)
        net_states = layers.Activation('relu')(net_states)

        # Add hidden layer(s) for action pathway

        # first block
        net_actions = layers.Dense(units=self.units_block_1, kernel_regularizer=regularizers.l2(self.l2))(actions)
        net_actions= layers.BatchNormalization()(net_actions)
        net_actions = layers.Activation('relu')(net_actions)
        
        # second block
        net_actions = layers.Dense(units=self.units_block_2, kernel_regularizer=regularizers.l2(self.l2))(actions)
        net_actions = layers.BatchNormalization()(net_actions)
        net_actions = layers.Activation('relu')(net_actions)

        # Combine state and action pathways
        net = layers.Add()([net_states, net_actions])
        net = layers.Activation('relu')(net)

        # Add more layers to the combined network if needed
        net = layers.Dense(units=self.units_block_3)(net)
        net = layers.BatchNormalization()(net)
        net = layers.Activation('relu')(net)

        # Add final output layer to prduce action values (Q values)
        Q_values = layers.Dense(units=1, name='q_values')(net)

        # Create Keras model
        self.model = models.Model(inputs=[states, actions], outputs=Q_values)

        # Define optimizer and compile model for training with built-in loss function
        # set learning rate to 10**-3 (0.001) as recommend by research paper
        optimizer = optimizers.Adam(lr=self.lr)  
        self.model.compile(optimizer=optimizer, loss='mse')

        # Compute action gradients (derivative of Q values w.r.t. to actions)
        action_gradients = K.gradients(Q_values, actions)

        # Define an additional function to fetch action gradients (to be used by actor model)
        self.get_action_gradients = K.function(
            inputs=[*self.model.input, K.learning_phase()],
            outputs=action_gradients)