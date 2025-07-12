
import numpy as np

class CategoricalNaiveBayes():
    """
    A simple implementation of the Categorical Naive Bayes classifier for discrete/categorical feature data.

    This model assumes that each feature follows a multinomial distribution conditioned on the class label.
    It applies Laplace smoothing controlled by the hyperparameter `alpha`.

    Attributes:
        X (ndarray): The training feature matrix of shape (n_samples, n_features).
        y (ndarray): The target labels corresponding to `X`, of shape (n_samples,).
        alpha (float): Laplace smoothing parameter (must be >= 0).
    """

    def __init__(self, alpha):
        """Constructor method. Class variables are stored or computed."""

        # Creating class variables,
        self.X, self.y = None, None
        self.nX_possible_values, self.y_possible_values = None, None
        self.alpha = alpha
        
        # Placeholders,
        self.Py_vector = None
        self.likelihoods = []

    def fit(self, X, y):
        """Bulk of the required calculation is performed by fitting the model. Specifically, our probability vectors are 
        computed."""

        # Assigning class variables,
        self.X, self.y = X, y

        # Computing possible values for features and classes,
        self.nX_possible_values = np.apply_along_axis(lambda col: len(np.unique(col)), axis=0, arr=X)
        self.n_features = X.shape[1]
        self.feature_values = [np.unique(X[:, i]) for i in range(self.n_features)]

        # Computing P(y) for all class labels,
        self.y_possible_values, y_counts = np.unique(y, return_counts=True)
        self.Py_vector = y_counts/len(y) # <-- Formula 
        self.n_classes = len(self.y_possible_values)
        self.max_categories = max(self.nX_possible_values)

        # Creating likelihood tensor,
        self.likelihoods = np.full((self.n_classes, self.n_features, self.max_categories), fill_value=1, dtype=float)

        # Computing all P(x_i|y=yk) for lik,
        for cls_idx, cls in enumerate(self.y_possible_values):

            cls_idxs = np.where(self.y==cls)[0]
            X_given_y = X[cls_idxs]

            # Double loop over features and then possible feature values,
            for feature_idx in range(self.n_features):

                # P(x_i|y=yk) for all values x_i can take,
                for feature_value in self.feature_values[feature_idx]:
                    n_instances = len(np.where(X_given_y.T[feature_idx]==feature_value)[0])
                    n_j = self.nX_possible_values[feature_idx]
                    Pxiy = (n_instances + self.alpha)/(len(cls_idxs) + n_j*self.alpha)
                    
                    # Adding to likelihood tensor,
                    self.likelihoods[cls_idx, feature_idx, feature_value] = Pxiy

    def predict(self, X_sample, return_probs=False):

        # Initialising arrays,
        self.Log_probs = []
        self.probs = None

        for class_idx, cls in enumerate(self.y_possible_values):

            # Extracting required P(y=y_l),
            Log_Py = np.log(self.Py_vector[class_idx])

            # Computing P(X|y=y_l),
            Log_PXy = 0 # <-- Placeholder value

            for feature_idx, feature_value in enumerate(X_sample):
                Log_PXiy = np.log(self.likelihoods[class_idx][feature_idx][feature_value]) # <-- Extracted from pre-computed matrix
                Log_PXy += Log_PXiy

            # Final calculation,
            Log_PykX = Log_Py + Log_PXy
            self.Log_probs.append(Log_PykX)

        if return_probs:

            # Recovering PXy_vector,
            PXy_vector = np.exp(self.Log_probs)
            
            # Computing P(X),
            PX = np.dot(self.Py_vector, PXy_vector)

            # Calculating probabilities, 
            self.probs = (self.Py_vector*PXy_vector)/PX

            return self.probs
        else:
            # Prediction as the most likely probability,
            pred = self.y_possible_values[np.argmax(self.Log_probs)]

            return pred

    def score(self, X, y):
        """The model prediction is taken as the class with the most likely probability."""

        # Counting number of correct predictions,
        correct = 0
        for i, X_sample in enumerate(X):
            pred = self.predict(X_sample)
            if pred == y[i]:
                correct += 1

        # Computing accuracy,
        accuracy = correct/len(y)

        return accuracy