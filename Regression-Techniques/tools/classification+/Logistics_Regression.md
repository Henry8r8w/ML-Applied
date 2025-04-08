## Binary Classification
**Goal**: To output `y`, a boolean (e.g, the binary) value classfication result, given a feature  `x`
- $x \rightarrow y,$ where  $x \in \mathbb{R}^{n_x}$ & $y \in {0,1}$

We define a `training set` with m samples as:

$$
\{(x^{(1)}, y^{(1)}), \dots, (x^{(m)}, y^{(m)})\}
$$
### Breaking The Training Set Down
X is an input matrix:

$$
X = \begin{bmatrix}
\vdots  &         &\vdots \\
x^{(1)} & \cdots  & x^{(m)} \\
\vdots  &         & \vdots
\end{bmatrix}
$$ 
- $X \in \mathbb{R}^{n_x \times m}$
- $X.\text{shape} = (n_x, m)$

Y is a label vector :

$$
Y = \begin{bmatrix}
y^{(1)} & \cdots & y^{(m)}
\end{bmatrix}
$$

- $Y \in \mathbb{R}^{1 \times m}$
- $Y.\text{shape} = (1, m)$

**Intuition**: There are `n_x features` per sample, and  `m samples` total. The more samples m we have, the better the model typically performs


**Notes**:
- You can think n_x as `rows` and m as `columns` if your brain got lazy, where each i in m is a training sample. Each **`y_i.shape = (1,1)`** is corresponding to a **`X_i.shape = (n_x, 1)`** into a training machine

## Logistics Regression
**Goal**: Given $x$, we want $\hat{y} = P(y = 1 | x)$
- $X \in \mathbb{R}^{n_x x 1}$
- Parameter accompied with $X$: $w \in \mathbb{R}^{n_x x 1}, b \in \mathbb{R}^{1 x 1}$

**Notes**:
- We are talking about 1 sample training set here

### Regression Expression

<p align="center"><strong>Linear Expression</strong></p>

$$\hat{y} = w^T X + b$$

**Notice**
- $\hat{y}$ is not binary bounded: $0 \leq \hat{y} \leq 1$
- $w^TX$ has a dimension of $(1, n_x) x (n_x, m) = (1, m)$
- $b$ has dimension of (1,m)
- **Note**: m is 1 in 1 sample training sample set speaking

<p align="center"><strong>Logistics Expression</strong></p>

$$\hat{y} = \sigma(w^T X + b);\ \sigma(z) = 1/ (1 + e^{-z})$$

**Notice**
- As $\hat{y} \rightarrow \infty, \sigma(z) \rightarrow 1$
- As $\hat{y} \rightarrow -\infty, \sigma(z) \rightarrow 0$


```python
import matplotlib.pyplot as plt
import numpy as np

def sigma(z):
    return 1 / (1 + np.exp(-z))

def linear_model(x):
    return 2 * x + 1  # linear model: z = wx + b ; dim -> (len(x), 1)

x = np.linspace(-10, 10, 100)
z = linear_model(x)
y = sigma(z)

plt.plot(x, y, label='Sigmoid Output')
plt.xlabel('x')
plt.ylabel('Probability')
plt.title('Logistic Regression: Sigmoid Function')
plt.grid(True)
plt.show()
```

## Cost Function

**Goal**: Given a training set, we want to have each $ \ \hat{y}^i \approx y^i$, where $y^i$ is the `test` value  and $\hat{y}^i$ is model `trained` predicted value

Recall, training set is defined as 
$$
\{(x^{(1)}, y^{(1)}), \dots, (x^{(m)}, y^{(m)})\}
$$

**`Loss function`** for logistics regression: 

$$
 L(\hat{y}, y) = -(y \log(\hat{y}) + (1-y) \log(1- \hat{y}))
$$

**Notice**
- At $y = 1: - \log(\hat{y}) = L$
  - Thus, we want $\hat{y} \rightarrow 1: L \rightarrow 0$
- At $y = 0: - \log(1- \hat{y}) = L$
  - Thus, we want $\hat{y} \rightarrow 0: L \rightarrow 0$



**`Average Loss Function`** for m-sample logistics regression

$$
 J(w, b) = \frac{1}{m} \sum^m_{i = 1} L(\hat{y}^i, y^i) = -(y \log(\hat{y}) + (1-y) \log(1- \hat{y}))
$$

**Notes**:
- Your train and test should all go into you machine to perform the MSE caluclation when in practice
- J(w,b) is in expression fo w and b, becuase y is in expression fo w and b. Sigmoid activation function simply limits the value into 0 and 1, but it define the loss here*

```
notations:
     1. *: uncertain, may be disproved

```
