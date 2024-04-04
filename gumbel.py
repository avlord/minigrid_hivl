from jax import numpy as jnp
import jax
import numpy as np

def sample_gumbel(shape, eps=1e-20): 
  """Sample from Gumbel(0, 1)"""
  U = np.random.uniform(0,1,shape)
  return jnp.log(-jnp.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature): 
  """ Draw a sample from the Gumbel-Softmax distribution"""
  y = logits + sample_gumbel(jnp.shape(logits))
  return jax.nn.softmax( y / temperature)

def gumbel_softmax(logits, temperature, hard=False):
  """Sample from the Gumbel-Softmax distribution and optionally discretize.
  Args:
    logits: [batch_size, n_class] unnormalized log-probs
    temperature: non-negative scalar
    hard: if True, take argmax, but differentiate w.r.t. soft sample y
  Returns:
    [batch_size, n_class] sample from the Gumbel-Softmax distribution.
    If hard=True, then the returned sample will be one-hot, otherwise it will
    be a probabilitiy distribution that sums to 1 across classes
  """
  y = gumbel_softmax_sample(logits, temperature)
  if hard:
    k = jnp.shape(logits)[-1]
    #y_hard = tf.cast(tf.one_hot(tf.argmax(y,1),k), y.dtype)
    y_hard = jnp.equal(y,jnp.max(y))
    y_hard = jnp.astype(y_hard,y.dtype)
    # y_hard = jnp.cast(
      # jnp.equal(y,jnp.reduce_max(y,1,keep_dims=True)),y.dtype)
    y =  jax.lax.stop_gradient(y_hard - y) + y
    



  return y.sum()

a = (1,2)
print(sample_gumbel(a))
logits = jnp.array([0.5,0.6])
print(gumbel_softmax_sample(logits,0.5))
gumbel_grads = gumbel_softmax(logits,0.5,hard=True)
print(gumbel_grads)
print(jax.grad(gumbel_softmax)(logits,0.1,hard=True))