# TensorFlow Optimizers

## Files in this directory:
* `LICENSE`: Apache license. I'm borrowing a lot of code from the original TensorFlow implementation so I'm pretty sure I need this in here.
* `optimizer.py`: Contains my own hacked-up version of the original Optimizer class in TensorFlow. Includes support for initialization bias corrected moving averages (dense and sparse).
* `base.py`: Contains the BaseOptimizer class. As the name suggests, the other optimizers build off this one. Contains the following hyperparameters:
  - `lr`: the learning rate. Set to `None` to use the Oren-Luerenburg scalar Hessian approximation as the learning rate (default is 1)
  - `eps`: a stability constant (default is 1e-16).
  - `chi`: the decay constant for temporal averaging (default is 0).
  - `clip`: updates can have at most this value for their L2 norm. (default is None)
  - `noise`: std dev of stochastic noise added to every update (default is 0).
  - `save_step`: whether to save the steps to a slot (needed for some Hessian approximations that ultimately didn't work for me; default is False)
  - `save_grad`: whether to save the grads to a slot (needed for some Hessian approximations that ultimately didn't work for me; default is False)
* `adam.py`: Vanilla AdamOptimizer. Adds the following hyperparameters:
  - `mu`: the momentum decay constant (default is .9)
  - `ups`: the uncentered variance decay constant (default is .9--.999 is WAY too high)
* `nadam.py`: NadamOptimizer that attempts to imitate NAG in Adam as closely as possible.
* `radam.py`: RadamOptimizer that gives more weight to the current gradient update without significantly crippling the optimizer's ability to remember past gradients. Adds the following hyperparameter:
  - `gamma`: the interpolation factor (default is .05)

## Algorithms
* SGD with OL learning rate:
```python
g_t = grads(loss, x_tm1)
y_t = g_t - g_tm1
lr = dot(s_tm1, s_tm1) / dot(s_tm1, y_t)
s_t = -abs(lr) * g_t
x_t = x_tm1 + s_t
```

* Adam
```python
g_t = grads(loss, x_tm1)
mu_t = mu * (1-mu**(t-1)) / (1-mu**t)
ups_t = ups * (1-ups**(t-1)) / (1-ups**t)
m_t = mu_t*m_tm1 + (1-mu_t)*g_t
v_t = ups_t*v_tm1 + (1-ups_t)*g_t**2
v_bar_t = sqrt(v_t + eps)
x_t = x_tm1 - lr*m_t / v_bar_t
```
* Nadam
```python
g_t = grads(loss, x_tm1)
mu_t = mu * (1-mu**(t-1)) / (1-mu**t)
mu_tp1 = mu * (1-mu**t) / (1-mu**(t+1))
ups_t = ups * (1-ups**(t-1)) / (1-ups**t)
m_t = mu_t*m_tm1 + (1-mu_t)*g_t
m_bar_t = mu_tp1*m_t + (1-mu_t)*g_t
v_t = ups_t*v_tm1 + (1-ups_t)*g_t**2
v_bar_t = sqrt(v_t + eps)
x_t = x_tm1 - lr*m_bar_t / v_bar_t
```
* Radam
```python
g_t = grads(loss, x_tm1)
mu_t = mu * (1-mu**(t-1)) / (1-mu**t)
ups_t = ups * (1-ups**(t-1)) / (1-ups**t)
m_t = mu_t*m_tm1 + (1-mu_t)*g_t
m_bar_t = (1-gamma)*m_t + gamma*g_t
v_t = ups_t*v_tm1 + (1-ups_t)*g_t**2
v_bar_t = sqrt(v_t + eps)
x_t = x_tm1 - lr*m_bar_t / v_bar_t
```