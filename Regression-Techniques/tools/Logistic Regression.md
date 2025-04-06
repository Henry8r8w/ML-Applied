# Logistic Regression

## The Idea
- Binary classification algorithm
- Predicts probability using sigmoid function: $\sigma(z) = \frac{1}{1 + e^{-z}}$
- Linear combination inside sigmoid: $z = \beta_0 + \beta_1 x_1 + ... + \beta_n x_n$
- Prediction: class 1 if probability ≥ 0.5, else class 0

## Workflow
```
Initialize weights w and bias b to 0 

For number_of_iterations:
    // Forward Propagation
    z = w⋅x + b  // linear combination
    a = sigmoid(z)                 
    
    // logitiscs loss
    loss = -1/m * sum(y*log(a) + (1-y)*log(1-a))
    
    // Backward Propagation
    dz = a - y                    
    dw = 1/m * X^T * dz          
    db = 1/m * sum(dz)              
    
    // Update weights and bias
    w = w - learning_rate * dw     
    b = b - learning_rate * db     

z = w⋅x + b
a = sigmoid(z)
prediction = 1 if a >= 0.5 else 0
```
## Implementation

### Raw Python Implementation
```python
class LogisticRegression:
    def __init__(self, lr=0.01, iterations=1000):
        self.lr = lr
        self.iterations = iterations
        self.weights = None
        self.bias = None
        
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent: 
        for _ in range(self.iterations):
            # Linear model + sigmoid
            y_pred = self.sigmoid(np.dot(X, self.weights) + self.bias)
            
            # Compute gradients
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)
            
            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
    
    def predict_prob(self, X):
        return self.sigmoid(np.dot(X, self.weights) + self.bias)
    
    def predict(self, X, threshold=0.5):
        return self.predict_prob(X) >= threshold
```


## CheatSheet
### Pre-req: Gradient Descent
**Forward and backward propagation for logistic regression**

Output Function: $\hat{y} = \delta(w^{T}x + b) = \delta(z)$

Sigmoid Function Definition: $\delta(z) = 1/ (1+e ^{z})$

**Forward**:

$$(x_1, w_1, x_2, w_2, b) \rightarrow z = w_1x_1+w_2x_2+b \rightarrow \hat{y} = a = \sigma(z) \rightarrow L(a,y)$$

**Backward**:
- $dL/da = \frac{-y}{a} + \frac{1-y}{1-a}$
- $dL/dz = dL/da \cdot da/dz = a - y$
- $dL/dw_1 = dL/dz \cdot dz/dw_1 = x_1(a - y)$
- $dL/dw_2 = dL/dz \cdot dz/dw_2 = x_2(a - y)$
- $dL/db = dL/dz \cdot dz/db = a - y$

**Update rules**:
- $w_1 = w_1 - \alpha \cdot dL/dw_1$
- $w_2 = w_2 - \alpha \cdot dL/dw_2$
- $b = b - \alpha \cdot dL/db$


### Pre-req: Logistics Loss Function
**Loss Function (single error)**
- $L(\hat{y}, y) = -(y\log(\hat{y})) + (1-y)\log(1-\hat{y})$
	- at y = 1, $L(\hat{y}, y) = -y\log(\hat{y})$
	- at y = 0, $L(\hat{y}, y) = (1-y)\log(1-\hat{y})$
- we want higher y_hat (upper bound of 1) for y = 1 and lower y_hat (lower bound of 1) for y = 0
	- with cat, the higher of predicted y to 1 / y_hat converge toward to 1, the lower the loss
	- with not(cat), the lower of the predicted y to 1 / y_hat converge toward to 0, the lower the loss
