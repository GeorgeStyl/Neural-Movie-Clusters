
import numpy as np


class NeuralNet:

    def __init__(self, feature_matrix, distance_matrix, hidden_nodes=25, k=40):
        self.k = k
        self.feature_matrix = feature_matrix
        self.distance_matrix = distance_matrix

        data_dim = self.feature_matrix.shape[1]
        self.hidden_nodes = hidden_nodes

        self.W1 = np.random.randn(data_dim, hidden_nodes) * np.sqrt(1 / data_dim)
        self.W2 = np.random.randn(hidden_nodes, data_dim) * np.sqrt(1 / hidden_nodes)

        self.b1 = np.zeros(hidden_nodes)
        self.b2 = np.zeros(data_dim)

        self.precompute_input_vectors(valid_indices=None)


    @staticmethod
    def compute_nearest_neighbors(user_ind, distance_matrix, k, valid_indices=None):
        """
        Returns the k nearest neighbors of a user, filtered only from those belonging in the training set.
        :param user_ind: Index of user in the feature matrix
        :param distance_matrix: NxN distance matrix (Jaccard)
        :param k: number of nearest neighbors to return
        :param valid_indices: indices of users that belong in the training set
        :return: numpy array of k nearest neighbors' indices
        """

        # 1) Get the array distances of the user with each other user
        # 2) Filter out users that belong in test set
        # 3) Sort the array based on distance, ascending
        # 4) Get the k first distances after the first (which will be 0 - distance to themself)

        distances = distance_matrix[user_ind, :]

        if valid_indices is not None:

            # Mask distances to only consider valid_indices
            mask = np.ones_like(distances, dtype=bool)
            mask[:] = False
            mask[valid_indices] = True
            # Set distances to test users to inf
            distances = np.where(mask, distances, np.inf)

        return distances.argsort()[1:k + 1]


    def precompute_input_vectors(self, valid_indices):
        """
        Precompute input vectors for all users. Input vectors are the averaged ratings from each user's k nearest
        neighbors.
        Called once during initialization. Results are stored in self.precomputed_inputs
        :return:
        """

        self.precomputed_inputs = {}

        for user_ind in range(len(self.feature_matrix)):
            neighbors = self.compute_nearest_neighbors(user_ind, self.distance_matrix, self.k, valid_indices)
            input_vector = self.create_input_layer(neighbors, self.feature_matrix)
            self.precomputed_inputs[user_ind] = input_vector


    @staticmethod
    def create_input_layer(neighbors_inds, feature_matrix):
        """
        Build the input vector for a user based on their k nearest neighbors (vectorized).
        :param neighbors_inds: indices of the k nearest neighbors of the target user
        :param feature_matrix: full feature matrix (users x movies)
        :return: 1D numpy array of length = number of features
        """

        # Extract neighbors features
        neighbors_matrix = feature_matrix[neighbors_inds, :]

        # Create mask of non-zero ratings
        nz_mask = neighbors_matrix > 0

        # Sum ratings per movie and divide by number of neighbors who rated
        sum_ratings = np.sum(neighbors_matrix, axis=0)
        num_ratings = np.sum(nz_mask, axis=0)

        # Avoid division by zero
        num_ratings[num_ratings == 0] = 1

        # Compute averaged feature vector
        input_vector = sum_ratings / num_ratings

        return input_vector


    def forward_prop(self, user_ind):
        """
        Forward propagation through the network. Computes predicted ratings for all movies for a given user.
        :param user_ind: row-index of user in the feature matrix
        :return: x=input vector, a=hidden nodes after activation, y=prediction
        """

        # Input layer
        x = self.precomputed_inputs[user_ind]

        # Hidden layer
        z = np.dot(x, self.W1) + self.b1
        a = np.tanh(z)

        # Output layer
        y = np.dot(a, self.W2) + self.b2

        return x, a, y


    def back_prop(self, user_ind, x, a, y, penalty=1):
        """
        Backward propagation to compute gradients, with respect to all parameters.
        """

        # Set target
        R_u = self.feature_matrix[user_ind, :]

        # Create mask to find rated movies
        rated_mask = R_u > 0

        # Get dimensions of target vector (only rated movies)
        n_rated = rated_mask.sum()

        # In the environment of our problem this won't ever be successful, but for completeness
        if n_rated == 0:
            return None

        # Define cost function: Averaged sum of squared errors (for rated movies, condition excluded in the formulas below)
        # In matrix form:
        # C = 1/n * (y-R)^2, with:
        # y = W2.T * a + b2
        # a = tanh(z)
        # z = W1.T * x + b1

        # Error only on rated movies
        error = np.zeros_like(y)
        error[rated_mask] = y[rated_mask] - R_u[rated_mask]

        # Derivatives
        dC_dy = (2 / n_rated) * error
        dC_db2 = dC_dy
        dC_dW2 = np.outer(a, dC_dy)
        dC_da = np.dot(dC_dy, self.W2.T)
        dC_dz = dC_da * (1 - a**2)
        dC_db1 = dC_dz
        dC_dW1 = np.outer(x, dC_dz)

        return dC_dW1, dC_db1, dC_dW2, dC_db2


    def update_weights(self, dC_dW1, dC_db1, dC_dW2, dC_db2, learning_rate=0.01, lambda_reg=0.001):
        """
        Use gradient descent to update the weights of the model
        """
        self.W1 -= learning_rate * (dC_dW1 + lambda_reg * self.W1)
        self.b1 -= learning_rate * dC_db1
        self.W2 -= learning_rate * (dC_dW2 + lambda_reg * self.W2)
        self.b2 -= learning_rate * dC_db2


    def train(self, epochs=200, tol=1e-3, valid_indices=None):

        num_users = len(valid_indices)
        prev_loss = np.inf
        best_loss = np.inf

        # Save best weights
        best_W1, best_W2 = self.W1.copy(), self.W2.copy()
        best_b1, best_b2 = self.b1.copy(), self.b2.copy()

        for epoch in range(epochs):

            total_loss = 0

            for user_ind in valid_indices:

                x, a, y = self.forward_prop(user_ind)

                R_u = self.feature_matrix[user_ind, :]

                rated_mask = R_u > 0

                # Loss only on rated movies
                n_rated = rated_mask.sum()
                if n_rated == 0:
                    continue

                error = y[rated_mask] - R_u[rated_mask]
                loss = np.sum(error ** 2) / n_rated
                total_loss += loss

                dw1, db1, dw2, db2 = self.back_prop(user_ind, x, a, y)

                self.update_weights(dw1, db1, dw2, db2)

            avg_loss = total_loss / num_users

            # Show progress at the start (to see gradient descent), then every 20 epochs
            if epoch < 10 or epoch % 20 == 19:
                print(f'Epoch {epoch + 1}/{epochs}, MSE: {avg_loss:.4f}')

            # Track best weights so far
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_W1, best_W2 = self.W1.copy(), self.W2.copy()
                best_b1, best_b2 = self.b1.copy(), self.b2.copy()

            # Stop condition
            if np.abs(prev_loss - avg_loss) < tol:
                print(f'Stopping early at epoch {epoch + 1} (Delta_loss < {tol})')
                break

            prev_loss = avg_loss

        # Restore best weights
        self.W1, self.W2 = best_W1, best_W2
        self.b1, self.b2 = best_b1, best_b2
        print(f'\nBest training loss: {best_loss:.4f}\n')


    def predict(self, user_ind, feature_matrix, distance_matrix, valid_indices):

        # Get nearest neighbors
        neighbors = self.compute_nearest_neighbors(user_ind, distance_matrix, self.k, valid_indices)

        # Create input layer
        x = self.create_input_layer(neighbors, feature_matrix)

        # Forward pass
        z = np.dot(x, self.W1) + self.b1
        a = np.tanh(z)
        y = np.dot(a, self.W2) + self.b2

        # Clip to valid rating range [0,10] and round
        y = np.clip(y, 0, 10)
        y = np.round(y)  # Also tested without rounding. Improvement in two runs is minimal (+ ~0.02 mae points compared
                         # to when we do round), but we lose the "exact predictions" metric, which naturally is 0% if
                         # we don't round

        return y