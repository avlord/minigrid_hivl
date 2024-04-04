import equinox as eqx
import jax.numpy as jnp
import optax
import jax
@eqx.filter_value_and_grad(has_aux=True)
def loss_fun(model,obs,y):
    x,mask = obs
    y_hat = model(x,mask)
    logits = jax.nn.log_softmax(y_hat)
    y_ohe = jax.nn.one_hot(y,10)
    loss= -(y_ohe*logits).sum()
    aux = jnp.argmax(logits)
    return loss, aux

def init_optimizer(model,lr=1e-1):
    optim = optax.adam(lr)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))
    return optim, opt_state

def update_model(model,grads,optim,opt_state):
    updates, opt_state = optim.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, (optim, opt_state)

@eqx.filter_value_and_grad
def compute_loss_pi(model,mask,obs,adv,):
    
    log_p = model(obs,mask)
    idx = jnp.argmax(log_p)
    log_p = log_p[idx]
    loss_pi = -log_p.reshape(-1)*adv
    return loss_pi.mean()

