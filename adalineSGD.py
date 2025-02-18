import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class AdalineSGD:
    """ADAptive LInear NEuron classifier.
    
    Parameters
    -----------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset.
    shuffle : bool (default: True)
        Shuffles training data every epoch if True to prevent 
        cycles.
    random_state : int
        Random number generator seed for random weight 
        initialization.
    Attributes
    ----------
    w_ : 1d-array
        Weights after fitting.
    b_ : Scalar
        Bias unit after fitting.
    losses_ : list
        Mean squared error loss function value averaged over all
        training examples in each epoch.
    """
    def __init__(self, eta=0.01, n_iter=10,
                    shuffle=True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        self.random_state = random_state

    def _shuffle(self, X, y):
        """Shuffle training data"""
        r = self.rgen.permutation(len(y))
        return X[r], y[r]


    def _initialize_weights(self, m):
        """Initialize weights to small random numbers"""
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01,
                                        size=m)
        self.b_ = np.float_(0.)
        self.w_initialized = True

    def _update_weights(self, xi, target):
        """Apply Adaline learning rule to update the weights"""
        output = self.activation(self.net_input(xi))
        error = (target - output)
        self.w_ += self.eta * 2.0 * xi * (error)
        self.b_ += self.eta * 2.0 * error
        loss = error**2
        return loss

    def fit(self, X, y):
        """ Fit training data.
            Parameters
            ---------
            X : {array-like}, shape = [n_examples, n_features]
                Training vectors, where n_examples is the number of 
                examples and n_features is the number of features.
            y : array-like, shape = [n_examples]
                Target values.
            Returns
            ------
            self : object
        """
        self._initialize_weights(X.shape[1])
        self.losses_ = []
        for i in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            losses = []
            for xi, target in zip(X, y):
                losses.append(self._update_weights(xi, target))
            avg_loss = np.mean(losses) 
            self.losses_.append(avg_loss)    
        return self
    

    def batchRun(self, X, y):
        """Fit one batch
        
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
            Training vectors, where n_examples is the number of examples and n_features is the number of features.
        y : array-like, shape = [n_examples]
            Target values.
        
        Returns
        -------
        self : object
        """
        #Does not reset weights. Meant to be used with mini batch SGD
        
        net_input = self.net_input(X)
        output = self.activation(net_input)
        errors = (y - output)
        self.w_ += self.eta * 2.0 * X.T.dot(errors) / X.shape[0]
        self.b_ += self.eta * 2.0 * errors.mean()
        loss = (errors**2).mean()
        
        #return loss in batch
        return loss
    
    def makeBatches(self, X, y, batch_size):
        #divide data into batches
        rows = X.shape[0]
        indices = np.random.choice(rows, rows, replace=False) 
        batches = []
        for i in range(0, rows, batch_size):
            start = i
            end = min(i+batch_size, rows)
            index = indices[start:end]
            batches.append((X[index], y[index]))
        return batches
    


    def fit_mini_batch_SGD(self, X, y, batch_size=1):
        #takes samples with features X = [n_samples, n_features]
        #batch size is one unless user specified.
        #chooses samples at random.
        self._initialize_weights(X.shape[1])

        #check for a valid batch size
        bSize = batch_size
        rows = X.shape[0]
        if batch_size > rows:
            bSize = rows
        if batch_size < 1:
            bSize = 1

        self.losses_ = []
        for x in range(self.n_iter):
            #shuffle for each epoch
            if self.shuffle:
                X, y = self._shuffle(X, y) 
            batches = self.makeBatches(X, y, bSize)
            bloss = []      
            for batch in batches:
                bloss.append(self.batchRun(batch[0], batch[1]))
            self.losses_.append(np.mean(bloss))    
         

        return self                   







    def partial_fit(self, X, y):
        """Fit training data without reinitializing the weights"""
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_) + self.b_

    def activation(self, X):
        """Compute linear activation"""
        return X

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)
    
def plot_decision_regions(X, y, classifier, resolution=0.02):
 # setup marker generator and color map
    markers = ('o', 's', '^', 'v', '<')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
 # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    lab = lab.reshape(xx1.shape)
    plt.figure()
    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                 y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=f'Class {cl}',
                    edgecolor='black')    

if __name__ == "__main__":
    # Load Iris dataset
    iris = datasets.load_iris()
    X = iris.data[:100, :]
    y = iris.target[:100]

    # Convert labels to (-1,1)
    #y = np.where(y == 0, -1, 1)

    # Define parameters
    eta = 0.01
    n_iter = 50

    # Train Adaline
    adaline = AdalineSGD(eta=eta, n_iter=n_iter, random_state=1)
    adaline.fit(X, y)

    # Plot Adaline Loss
    plt.figure()
    plt.plot(range(1, len(adaline.losses_) + 1), adaline.losses_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error')
    plt.title('AdalineSGD Loss Over Epochs')


    adaline2 = AdalineSGD(eta=eta, n_iter=n_iter, random_state=1)
    adaline2.fit_mini_batch_SGD(X, y, 3)

    # Plot Adaline Loss
    plt.figure()
    plt.plot(range(1, len(adaline2.losses_) + 1), adaline2.losses_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error')
    plt.title('Adaline_mini_batch_SGD Loss Over Epochs')

    adaline2 = AdalineSGD(eta=eta, n_iter=n_iter, random_state=1)
    adaline2.fit_mini_batch_SGD(X, y, X.shape[0])

    # Plot Adaline Loss
    plt.figure()
    plt.plot(range(1, len(adaline2.losses_) + 1), adaline2.losses_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error')
    plt.title('AdalineGD Loss Over Epochs')

    
    adaline = AdalineSGD(eta=eta, n_iter=n_iter, random_state=1)
    adaline.fit(X[:,[0,2]], y)
    plot_decision_regions(X[:, [0,2]], y, classifier=adaline)
    plt.xlabel('Sepal length [cm]')
    plt.ylabel('Petal length [cm]')
    plt.legend(loc='upper left')
    plt.title("AdalineSGD classification")

    adaline = AdalineSGD(eta=eta, n_iter=n_iter, random_state=1)
    adaline.fit_mini_batch_SGD(X[:,[0,2]], y, 3)
    plot_decision_regions(X[:, [0,2]], y, classifier=adaline)
    plt.xlabel('Sepal length [cm]')
    plt.ylabel('Petal length [cm]')
    plt.legend(loc='upper left')
    plt.title("AdalineSGD mini batch classification")

    adaline = AdalineSGD(eta=eta, n_iter=n_iter, random_state=1)
    adaline.fit_mini_batch_SGD(X[:,[0,2]], y, X.shape[0])
    plot_decision_regions(X[:, [0,2]], y, classifier=adaline)
    plt.xlabel('Sepal length [cm]')
    plt.ylabel('Petal length [cm]')
    plt.legend(loc='upper left')
    plt.title("AdalineGD classification")
    plt.show()



                 