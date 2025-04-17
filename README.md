# Machine Learning - Applied CheatSheet & Notes

### `Regression Techniques`
- **PreProcessing**: Missing Data, Categorical Data
- **Good Practices**: Pipelining, Cross-Validation
- **Tools**:
  - **Linear Models**: Linear, Elastic Net (Ridge, Lasso), Huber, Kernel Ridge  
  - **Tree-Based**: Random Forest, XGBoost  
  - **Ensemble Learning**: Boosting, Bagging, Stacking  
  - **Time Series**: ARIMA, LSTM, Prophet (FB)  
  - **Classification+**: Logistic Regression, Random Forest*, XGBoost*
    
Notes: + denotes intersection, * denotes duplicates

### `Deep Learning`
- **Methods**:
    - **Neural Networks** (logistic regression $\rightarrow$ Softmax, gradient descent $\rightarrow$ SGD, Adam)
      - Concepts: Forward/Backprop, Activation Functions
      - Toolkit: CNNs, RNNs, LSTMs, GRUs, Transformers
    - **Renforcement Learning**
      - Toolkit: Q-Learning, Deep Q Network, Policy Gradient Methods (REINFORCE, PPO, A3C)

- **Frameworks**: Pytroch (research & prototyping), Tensorflow (industry scale), JAX (autodiff & parallel computation)
  
### `Computer Visions`
- **Concepts**:
  - Image Classification
  - Object Detection, Segmentation
  - Data Augmentation, Transfer Learning
- **Framework**: OpenCV, YOLO, Detectron

### `Edge AI & Embedded Systems`
- **Hardware STM32** $\leftarrow$  C programming, GPIO, UART/SPI/I2C, RTOS
- **Framework**: 
    - **Based on STM32**:
        - CMSIS-NN (ARM optimized neural nets)
        - STM32Cube.AI
        - TensorFlow Lite for Microcontrollers (tflite-micro)
  
### `Causal Inference`
- Average Treatment Effect, Nearest Neighbor, Propensity Score Matching (PSM), IV: Two-Stage Least Squares (2SLS), DiD, Meta-learners (T-Learner, S-Learner, X-Learner)
  
### `Language Processing`
- LangChain

### `Agents`
- **Tools**: Generative Adversarial Networks (GANs), Markov Decision Processes (MDPs)
---
# Adjacent Topics to Machine Learning 

### `Signal Processing`

### `Probability`
