---
subtitle:    Brief Introduction to Deep Learning
chapter:     6
feedback:
  deck-id:  'deeprl-deep-learning'
...

------------------------------------------------------------------------------

# Content

------------------------------------------------------------------------------

# Content

::: small
::: columns-5-5
::: platzhalter
**Content**:

- The three main learning paradigms
- Supervised learning
  - The learning diagram
  - Generalization
  - Bias \& variance
  - Cross-validation
- Deep neural networks
  - Artificial neural networks
  - Multi-layer perceptron
  - Training
  - Regularization
  - Neural network architectures
  - Linear regression as a special case
:::

::: platzhalter
**Useful references**:

- Pattern recognition and machine learning [@Bishop2006prml]
- Deep learning [@bengio2017deep]
- The elements of statistical learning [@hastie2009elements]
- Learning from data: a short course [@Abu2012learning]
:::
:::
:::

------------------------------------------------------------------------------

# The three main learning paradigms

------------------------------------------------------------------------------

# The three main learning paradigms

![The three big learning paradigms [@Abdelwanis2026] (Examples: [Unsupervised](https://medium.com/analytics-vidhya/beginners-guide-to-unsupervised-learning-76a575c4e942), [Supervised](https://www.linkedin.com/pulse/what-supervised-learning-sahib-singh-ru3qc/), [RL](https://medium.com/analytics-vidhya/a-beginners-guide-to-reinforcement-learning-and-its-basic-implementation-from-scratch-2c0b5444cc49))](images/06-deep-learning/ML-paradigms.svg){ .embed width=1280px }

------------------------------------------------------------------------------

# Supervised learning

------------------------------------------------------------------------------

# The learning diagram

![The learning diagram [@Abu2012learning]](images/06-deep-learning/Learning-diagram.svg){ .embed width=800px }

# Generalization

::: small

::: incremental
- What does it mean to have a perfect model on your training dataset $\Dtrain$, i.e., $L(w, \Dtrain) = 0$?\
[$\Rightarrow$ we have simply memorized the data!]{.fragment}
- Learning means that we get **good predictions on unseen data**:
$$ \underbrace{L(w, \Dtest)}_{\text{in-sample error}} \approx \underbrace{L(w, \Dtrain)}_{\text{out-of-sample error}}. $$
:::

::: fragment
::: definition
### Supervised learning

Given a labeled (and probably noisy) dataset $\Dtrain=\{(x_1,y_1),\ldots,(x_N,y_N)\}$, approximate the unknown mapping $f : \Xc \to \Yc$ by a parametrizable ML model $f_\theta: \Xc \to \Yc$, such that $$ f_\theta(x_k) = \hat{y}_k \approx y_k \quad \forall ~ (x_k,y_k)\in\Dtest .$$
:::
:::

::: incremental
- **Goodness of fit** can be measured via many different metrics (e.g., mean squared error, classification accuracy, etc.).
- The dimension $d$ of model parameters $\theta \in\R^d$ is adjustable in many model families, which trades off **bias** with **variance** (among other factors, leading to so-called under- and overfitting).
- On top of $\theta$, an ML model might also have **hyperparameters** that can be optimized (e.g., number of layers in a neural network).
:::
:::

# Bias-variance tradeoff (1)

::: small
::: columns-7-4
::: platzhalter

[$\bullet$ In the ML context, **bias** denotes the error of the *average* model $\overline{f}_{\theta}$ when repeating the training with different datasets $\Dc_{\mathsf{train},1},\Dc_{\mathsf{train},2},\ldots$:
$$ \mathsf{bias} = \Expsub{\left(\overline{f}_{\theta}(x) - f(x)\right)^2}{x\sim\Dtest}. $$]{ .fragment data-fragment-index=1}

[$\bullet$ **variance** denotes the variability in between the individual training runs:
$$ \mathsf{variance} = \Expsub{\Expsub{\left(f^{(\Dc)}_{\theta}(x) - \overline{f}_{\theta}(x)\right)^2}{\Dc\sim\{\Dc_{\mathsf{train,\ell}}\}_{\ell=1}^\infty}}{x\sim\Dtest}. $$]{ .fragment data-fragment-index=2}

:::

[
$\Rightarrow$ Often a matter of model complexity.
![](images/06-deep-learning/BV-Bias_and_variance_contributing_to_total_error.svg){ .embed width=420px }
]{ .fragment data-fragment-index=7}

:::

::: columns-1-1-1-1
[![Low bias, low variance](images/06-deep-learning/BV-En_low_bias_low_variance.png){ .embed width=300px }]{ .fragment data-fragment-index=3}

[![Large bias, low variance](images/06-deep-learning/BV-Truen_bad_prec_ok.png){ .embed width=300px }]{.fragment data-fragment-index=4}

[![Low bias, large variance](images/06-deep-learning/BV-Truen_ok_prec_bad.png){ .embed width=300px }]{.fragment data-fragment-index=5}

[![Large bias, large variance [[Wikipedia](https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff)]](images/06-deep-learning/BV-Truen_bad_prec_bad.png){ .embed width=300px }]{.fragment data-fragment-index=6}
:::
:::

# Bias-variance tradeoff (2)
::: small
**Example** (see [Wikipedia](https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff)): Fitting a model of serveral radial basis functions to noisy trainig data: 
$$ f_\theta(x) = \sum_{k=1}^d \theta_k \exp\left(-\frac{1}{2}\frac{x-c_k}{\sigma_k^2}\right). $$

[$\bullet$ For a wide spread, the bias is high: the RBFs cannot fully approximate the function (especially the central dip), but the variance between different trials is low.]{ .fragment data-fragment-index=1}

[$\bullet$ As spread decreases (image 3 and 4) the bias decreases: the blue curves more closely approximate the red...]{ .fragment data-fragment-index=2}

[$\bullet$ ... but the variance between trials ($\Dc_{\mathsf{train,1}},\Dc_{\mathsf{train,2}},\ldots$) increases.]{ .fragment data-fragment-index=3}


:::

::: columns-1-1-1-1
![Function and training data $\Dc_{\mathsf{train,1}}$.](images/06-deep-learning/BV-Test_function_and_noisy_data.png){ .embed width=300px }

[![Wide spread RBFs.](images/06-deep-learning/BV-Radial_basis_function_fit,_spread=5.png){ .embed width=300px }]{.fragment  data-fragment-index=1 }

[![Medium spread RBFs.](images/06-deep-learning/BV-Radial_basis_function_fit,_spread=1.png){ .embed width=300px }]{.fragment  data-fragment-index=2 }

[![Small spread RBFs.](images/06-deep-learning/BV-Radial_basis_function_fit,_spread=0.1.png){ .embed width=300px }]{.fragment  data-fragment-index=3 }
:::

# Cross-validation

::: small
::: columns-5-7

![5-fold CV example [@Abdelwanis2026].](images/06-deep-learning/cross-validation.png){ .embed width=500px }

::: incremental
- Training is repeated $k$ times with $k$ different splits of the training set.
- Each observation serves as unseen instance (blue boxes) at least once.
- The validation error is an indicator for tuning hyperparameters.
- Example of a $k$-fold Cross-validation (CV).
:::
:::
:::

# Means to improve a supervised learning model

::: small
::: columns-7-4
::: incremental
- Collecting **more data**, i.e., increasing $N$.
- **Reducing noise** in the data.
- **Improving the distribution** within the dataset, i.e., ensuring that the data set is representative of the problem domain.
- **Choosing a more appropriate model**. A genereal rule is to select the model according to the amount of data one has, not according to the expected complexity of the funtion to approximate.
- **Optimizing hyperparameters** of the model.
- **Ensemble learning**: Averaging over several different models.
- **Including knowledge**, e.g., in the form of tailored features (feature engineering) or informed loss functions.
:::

::: fragment
![Euclidean vs. polar coordinates/features in binary classification [@Abdelwanis2026].](images/06-deep-learning/feature-engineering-linear.svg){ .embed width=280px }
:::
:::
:::

------------------------------------------------------------------------------

# Deep neural networks

------------------------------------------------------------------------------

# Artificial neural networks

::: small
Artificial neural networks (ANNs) are nonlinear function approximators $\hat{y}=f_\theta(x)$ that

::: incremental
- are end-to-end differentiable.
- are stacks of minimal units, the **artificial neurons**.
:::

\

::: fragment
::: columns-5-5
![An artificial neuron [@Abdelwanis2026].](images/06-deep-learning/neuron.svg){ width=400px }

::: incremental
- An ANN consists of **nodes** or **neurons** in one or more layers.
- Each node transforms the weighted sum of all previous nodes (plus a potential **bias term**) through an **activation function** $\sigma$:
$$ \sigma\left( \theta_0 + \sum_{k=1}^n \theta_k x_k \right). $$
- The weighted connections are called **edges**, which represent the ANN’s parameters.
:::

:::
:::
:::

# Multi-layer perceptron

::: small
Standard model of supervised learning: **multi-layer perceptron** or **feed-forward ANN**.

\

::: fragment
::: columns-4-6
![MLP architecture [@Abdelwanis2026].](images/06-deep-learning/MLP.svg){ width=450px }

::: incremental
- Only forward-flowing edges.
- The depth $L$ and width $H^{(\ell)}$ are hyperparameters.
- With $\sigma^{(\ell)}$ and $z^{(\ell)}$ denoting the activation function and **activation** of layer $\ell$ respectively, we get for the output in the $\th{\ell}$ layer.
$$ x^{(\ell)} = \sigma^{(\ell)}\big( \underbrace{\Theta^{(\ell)} x^{(\ell-1)} + b^{(\ell)}}_{z^{(\ell)}} \big) ,$$
with input $x^{(0)}=x$ and output $x^{(L)}=y$:
- Training:
  - Summarize the full set of parameters (i.e., weight matrix $\Theta^{(\ell)}\in\R^{H^{(\ell)} \times H^{(\ell-1)}}$ and biases $b^{(\ell)}$) under $\theta$.
  - Iteratively update the weights using gradient information.
:::
:::
:::
:::

# Activation functions

::: columns-4-1
::: incremental
- The source of nonlinearity in neural networks.$^*$
- Common choices for $\sigma(z)$ are
  - $\sigma(z) = \tanh(z)$,
  - Sigmoid: $\sigma(z) = \frac{1}{1+e^{-z}}$,
  - Rectified linear unit (ReLU): $\sigma(z) = \max(0, z)$,
:::

![Exemplary activation functions [@Abdelwanis2026].](images/06-deep-learning/activation.png){ width=370px }
:::

::: incremental
- The activation of the output layer, $\sigma^{(L)}(z)$, is task-dependent. For instance, 
  - regression: $y=\sigma^{(L)}(x^{(L-1)})=x^{(L-1)}$, i.e., $\sigma^{(L)} = \mathsf{Id}$ is the identity mapping.
  - binary classification: sigmoid (i.e., probability), followed by a rounding step to either $0$ or $1$.
  - multi-class classification: $y_i=\frac{\exp(z^{(L-1)}_i)}{\sum_j \exp(z^{(L-1)}_j)}$ (softmax).
:::

\ 

\ 

::: footer
$^*$ Without nonlinear activation functions, every ANN collapses to a single matrix-vector multiplication $y=\hat\Theta x$: 
$$x^{(\ell+2)}=\Theta^{(\ell+2)} x^{(\ell+1)} = \Theta^{(\ell+2)} \left(\Theta^{(\ell+1)} x^{(\ell)}\right)  = \left(\Theta^{(\ell+2)} \Theta^{(\ell+1)}\right)x^{(\ell)} = \hat\Theta x^{(\ell)} .$$
For non-zero biases, we obtain an *affine* transformation instead.
:::

# Training (1)

::: columns-7-3
::: incremental
- Training is performed in an iterative manner: $$\theta \gets \theta + \eta \delta\theta,$$ where $\eta\in\R_{>0}$ is the **step size** or **learning rate**, and $\delta\theta\in\R^d$ is the **update direction**, usually a gradient-based **descent direction**.
- First, we need to define a **loss function** that we wish to minimize.
  - Regression: (root) mean square error, mean absolute error.
  - Classification: cross entropy.
- Several iterations over the dataset $\Dtrain$ are called **depochs**.
:::

![The loss landscape of deep neural networks (with and without skip-connection) [@li2018visualizing].](images/06-deep-learning/loss-landscapes.png){ width=300px }
:::

# Training (2)

::: small
::: incremental
- The descent direction is computed by taking the derivative of the loss function w.r.t. the weight vector. In terms of the MSE:
[$$\begin{align*} 
\nabla L(\theta) &= \nabla \left(\frac{1}{N} \sum_{k=1}^N \|f_\theta(x_k) - y_k\|_2^2 \right) \\
&= \frac{1}{N} \sum_{k=1}^N \nabla \|f_\theta(x_k) - y_k\|_2^2 \\
&= \frac{1}{N} \sum_{k=1}^N \left(2 \|f_\theta(x_k) - y_k\|_2 \cdot \nabla f_\theta(x_k)\right).
\end{align*}$$]{.math-incremental}
- This means that we need to *propagate* the error through our model $f_\theta$.
- Since a neural network is a chain of neurons, we need to apply the **chain rule** of differentiation.
- As a consequence the loss is **backpropgated** through the network to determine the individual descent directions: $\frac{\partial L}{\partial\theta_i}$.
:::

::: fragment
::: definition
This is called the **backpropagation** algorithm. We require one forward pass and one backward pass for every data tuple $(x_k,y_k)$. Taking the average gives us the average steepest-descent improvement over the dataset $\Dtrain$ for the current $\theta$.
:::
:::
:::

# Regularization

# Neural network architectures

![The neural network zoo [[Asimov Institute](https://www.asimovinstitute.org/neural-network-zoo/)].](images/06-deep-learning/NN-zoo.png){ width=1280px }

# Linear regression as a special case

------------------------------------------------------------------------------

# Summary / what you have learned

------------------------------------------------------------------------------

# Summary / what you have learned

::: small
::: incremental
- a
:::
:::

# References

::: { #refs }
:::