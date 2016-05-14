# Optimization

## Overview of Repo
This repo contains a small handful of first-order optimizers, including Adam (which can be used as a momentum optimizer or an RMSProp optimizer), Nadam (which can be used as one kind of Nesterov optimizer or an RMSProp optimizer), and Radam (which can be used as another kind of Nesterov optimizer or an RMSProp optimizer). So far only TensorFlow is supported, but Lasagne implementations will probably come soon, and maybe more general Theano ones if I find time (Theano lacks some of the functionality that TensorFlow has, specifically `sparse_segment_sum`, making updating word embeddings tricky).

## Motivation for Nadam and Radam
This originally comes from a CS229 term paper and an ICLR workshop poster presentation, but the experiments in both were pretty preliminary so so far I haven't put anything on Arxiv. Below is the intuition I lay out in the papers, minus any experimental evidence or regret bounds. If you want to use one of the optimizers in your own work, please cite the ICLR presentation. Thanks!

### Why Adam is the coolest first order algorithm there is 
Adam basically combines correct versions of momentum and AdaGrad in the correct way. Using notation you'll find elsewhere in this repo, the classical version of momentum you often find in deep learning libraries has the following format:
```python
m_t = mu*m_tm1 + lr_t*g_t
x_t = x_tm1 - m_t  
```
where `x`, `g`, and `m` are the parameters, gradient, and momentum vector respectively, and `mu` and `lr` are the momentum decay factor and learning rate respectively. The problem with this formulation is that the expected step size depends on a lot of different factors--with SGD, the expected step size is `lr_t * E[|g_t|]`, which is great because we know the magnitude of the gradient is probably going to be relatively stable, so we can control how much the parameters change very straightforwardly with a schedule for the learning rate. With classical momentum, by contrast, the expected step size works out to be approximately `E[lr_t] * (1-mu**t) / (1-mu) * E[|g_t|]`. This is a mess: because `m` includes previous learning rates, the expected step size depends on all previous learning rates, not just the current one; because `m` is a decaying *sum* of the previous gradients, we have a factor of `1 / (1-mu)` that makes the expected step size larger, with how much larger it is being a function of `mu` (so different values of `mu` will result in different effective learning rates); and because `m_0` is initialized to zero, the magnitude of `m` grows as a function of `mu` and `t`, (so different values of `mu` will result in different effective annealing schedules). Adam fixes all of these issues by redefining momentum in the following way:
```python
m_t = mu*m_tm1 + (1-mu)*g_t
m_hat_t = m_t / (1-mu**t)
x_t = x_tm1 = lr_t * m_hat_t
```
This gives it an expected step size of `lr_t * E[|g_t|]`, the same as SGD--this is awesome for tuning hyperparameters, because it means the same learning rate and annealing schedule that work well for SGD will probably work well for momentum, no matter what `mu` might be.

AdaGrad has similar problems:
```python
v_t = v_tm1 + g_t**2
x_t = x_tm1 - g_t / sqrt(v_t + eps)
```
(Whether you put epsilon inside or outside the `sqrt` is generally inconsequential, but sometimes TensorFlow doesn't like taking the square root of zero)
The expected step size here is approximately `lr_t / sqrt(t) * E[|g_t / sqrt(E[g_t**2])|]`. While the last term looks kind of ugly, it's just the expected magnitude of the gradient normalized by its expected distance from zero--the factor of `1/sqrt(t)` is actually much more worrying, because it builds in an annealing schedule that researchers may not want. RMSProp fixes it a little bit by using a moving average of the denominator instead:
```python
v_t = ups*v_tm1 + (1-ups)*g_t**2
x_t = x_tm1 - g_t / sqrt(v_t + eps)
```
resulting in an expected step size of `lr_t / sqrt(1-ups**t) * E[|g_t / sqrt(E[g_t**2])|]`, where `ups` is the decay factor. The moving average gets rid of the `1 / sqrt(t)`, but introduces a `1 / sqrt(1-ups**t)` that can be pretty bad as well, resulting in higher effective learning rates in the first few iterations. These higher-than-normal effective learning rates can result in immediate divergeance if you don't use a lower learning rate--but the lower learning rate can cripple learning after the `ups**t` has basically reached zero. Adam fixes this as well, so that the expected step size is roughly `lr_t * E[|g_t / sqrt(E[g_t**2])|]` and only depends on the learning rate and properties of the gradient that we can assume will be relatively stable.
```python
v_t = ups*v_tm1 + (1-ups)*g_t**2
v_hat_t = v_t / (1-ups**t)
x_t = x_tm1 - g_t / sqrt(v_hat_t + eps)
```
Adam then takes these two algorithms and sticks them together, such that `x_t = x_tm1 - m_hat_t / sqrt(v_hat_t + eps)`, giving it the same expected step size as the corrected version of AdaGrad. Nice!

You can also do initialization bias correction with a schedule on `mu` and `ups` such that:
```python
mu_t = mu * (1-mu**(t-1)) / (1-mu**t)
ups_t = ups * (1-ups**(t-1)) / (1-ups**t)
```
This avoids the need for `m_hat` and `v_hat` because `m` and `v` contain the initialization bias corrected values.

One final thing to note is that in order to maintain the nice step size properties of the algorithm when you train word embeddings, you need to only decay the accumulators `m` and `v` for word embeddings that were actually used in the minibatch. Otherwise with momentum the step size shrinks and with rmsprop the step size grows as a function of the decay constant and word frequency. This is easy enough to do in TensorFlow but I haven't quite figured out how to do it in Theano yet.

### Incorporating Nesterov Momentum into Adam
The intuition behind Nesterov momentum is that `m_t` doesn't depend on the current gradient, so if we apply it *before* we do the gradient computation, we should get a higher-quality gradient. The way it's formulated in Sutskever et al. looks like basically like this (triple hash marks are on relevant lines):
```python
g_t = grads(loss, x_tm1 - mu*m_tm1) ###
m_t = mu*m_tm1 + lr*g_t
x_t = x_tm1 - m_t
```
But the way `g_t` is defined is awkward and difficult to implement. So let's simplify it a bit by first taking `mu*m_tm1` out of `g_t` and putting it in a different variable:
```python
x_hat_t = x_tm1 - mu*m_tm1 ###
g_t = grads(loss, x_hat_t) ###
m_t = mu*m_tm1 + lr*g_t
x_t = x_tm1 - m_t
```
But it turns out we subtract `mu*m_tm1` from the parameters `x` twice here--once in `x_hat_t`, then again in `x_t` (since `m_t` contains `mu*m_tm1` in it). Instead, we can define `x_t` in terms of `x_hat_t`, giving us the following:
```python
x_hat_t = x_tm1 - mu*m_tm1
g_t = grads(loss, x_hat_t)
m_t = mu*m_tm1 + lr*g_t
x_t = x_hat_t - lr*g_t ###
```
This is now a little weird too, because the algorithm starts and ends with independent updates to the parameters, so going from time `t` to time `t+1` involves two successive parameter updates--why not combine them into one parameter update? We can do this by subtracting `mu*m_tm1p1` at the definition of `x_t`, where `m_tm1p1` is `m_tm1` at the next timestep `t+1`, which is obviously `m_t` at the current timestep:
```python
g_t = grads(loss, x_t)
m_t = mu*m_tm1 + lr*g_t
x_t = x_t - mu*m_t - lr*g_t ###
```
But we can see that this is just the sum of a classical momentum update and an SGD update! Let's rewrite this a little bit for elegance:
```python
m_t = mu*m_tm1 + lr*g_t
m_bar_t = mu*m_t + lr*g_t ###
x_t = x_t - m_bar_t
```
And if we want to give it some of the nice expected step size properties that characterize Adam's momentum rule:
```python
m_t = mu*m_tm1 + (1-mu)*g_t
m_bar_t = mu*m_t + (1-mu)*g_t
x_t = x_t - lr*m_bar_t
```
However, there's one caveat--if we want to use a schedule for mu, which is the easiest way to do bias initialization correction, we have to make a slight adjustment to `m_bar_t`, using `mu_tp1`--the value of `mu_t` at the next timestep--because we're using the next timestep's momentum update instead of the current one's.
```python
m_t = mu_t*m_tm1 + (1-mu_t)*g_t
m_bar_t = mu_tp1*m_t + (1-mu_t)*g_t ###
x_t = x_t - lr*m_bar_t
```
If we combine this with Adam's AdaGrad/RMSProp component, we get Nesterov-accelerated Adaptive Moment Estimation, or 'Nadam'.

One question the above reformulation of Nesterov momentum raises is how much it matters whether or not you use `mu_tp1` in `m_bar_t`--why not just use `mu_t`? After all, it looks like `m_bar_t` is at its heart just interpolating between momentum and SGD, using `mu_t` as the interpolation factor. Maybe the benefit of Nesterov momentum that many researchers have reported comes from interpolating between SGD and momentum, not from applying the momentum part of the update before the gradient part. If that's so, then there's no reason to expect the optimal interpolation factor to depend on `mu` at all. So we can instead define an algorithm that interpolates between `m_t` and `g_t`, with interpolation factor `gamma`, effectively giving more weight to the current gradient than it would get under a classical momentum update.
```python
m_t = mu_t*m_tm1 + (1-mu_t)*g_t
m_bar_t = (1-gamma)*m_t + gamma*g_t ###
x_t = x_t - lr*m_bar_t
```
Here `gamma` decides how much more weight to give to `g_t`--good values seem to be somewhere between `.1` and `.01`, but I'm still working that out (for now I'm using `.05` in my own projects). When we combine this with Adam in the same was as we did with Nadam, we get 'Radam', or 'Reweighted Adaptive Moment Estimation'.

The post-hoc intuition for why reweighting should make a difference is that SGD assigns too much weight to `g_t`, allowing for oscillation, and momentum assigns too much weight to previous gradients in `m_t`, allowing for overshooting (this could also be remedied by decreasing `mu`, but that comes with the added consequence of forgetting older gradients faster, which might not always be desireable). Interpolating between the two should cancel out some of that error.
