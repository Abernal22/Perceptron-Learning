import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

class Perceptron:
    """Perceptron classifier.
    
    Parameters
    ------------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset.
    random_state : int
        Random number generator seed for random weight initialization.
    
    Attributes
    -----------
    w_ : 1d-array
        Weights after fitting.
    b_ : Scalar
        Bias unit after fitting.
    errors_ : list
        Number of misclassifications (updates) in each epoch.
    """
    
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
    
    def fit(self, X, y):
        """Fit training data.
        
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
            Training vectors, where n_examples is the number of
            examples and n_features is the number of features.
        y : array-like, shape = [n_examples]
            Target values.
        
        Returns
        -------
        self : object
        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1]+1)
        self.w_[:-1] = np.float64(0.)
        self.errors_ = []
        
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_ += update * np.append(xi, 1)
                errors += int(update != 0.0)
            self.errors_.append(errors)
        
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(np.append(X,1), self.w_)

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, 0)


# Load the full Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.2, random_state=42)

#Train three perceptrons
perceptrons = {}
classes = np.unique(y)

for c in classes:
    y_binary = np.where(y_train == c, 1, 0)
    perceptrons[c] = Perceptron(eta= 0.01, n_iter=100)
    perceptrons[c].fit(X_train, y_binary)

#Multiclass Prediction using OvA
def predict_multiclass(X):
    scores = {c: perceptrons[c].net_input(X) for c in classes}
    return max(scores, key=scores.get)

y_pred = np.array([predict_multiclass(xi) for xi in X_test])

accuracy = accuracy_score(y_test, y_pred)
print(f"Multiclass Perceptron Accuracy: {accuracy:.2f}")

# Plot Perceptron Errors for all Classes
plt. figure(figsize=(8, 6))
for c in classes:
    plt.plot(range(1, len(perceptrons[c].errors_) + 1), perceptrons[c].errors_, marker='o', label=f'Class {c}')

plt.plot(range(1, len(perceptrons[classes[0]].errors_) + 1 ), perceptrons [classes[0]].errors_, marker= 'o')
plt.xlabel('Epochs')
plt.ylabel('Number of Updates')
plt.title('Perceptron Convergence for all Classes')
plt.show()


if __name__ == "__main__":
  # Load Iris dataset
  X_binary = iris.data[:100, :]
  y_binary = iris.target[:100]

  # Convert labels to (-1,1)
  #y = np.where(y == 0, -1, 1)

  # Define parameters
  eta = 0.1
  n_iter = 50

  # Train Perceptron
  perceptron = Perceptron(eta=eta, n_iter=n_iter)
  perceptron.fit(X_binary, y_binary)

  # Plot Perceptron Errors
  plt.plot(range(1, len(perceptron.errors_) + 1), perceptron.errors_, marker='o')
  plt.xlabel('Epochs')
  plt.ylabel('Number of Updates')
  plt.title('Perceptron Convergence Over Epochs')
  plt.show()
