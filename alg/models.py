from typing import Any
import equinox as eqx
import jax
import jax.numpy as jnp


class Model(eqx.Module):
    sub_models: list
    obs: list

    def __init__(self) -> None:
        self.sub_models = []
        self.obs = []
    
    def add_model(self,model):
        self.sub_models.append(model)

    def step(self,obs):
        pred = self(obs)
        return jnp.argmax(pred), pred
    
    def update(self,):
        pass    
    
    def add_training_example(self,obs):
        self.obs.append(obs)

class PGModel(Model):
    layers:list
    mask: jnp.array
    def __init__(self,out_size=10,mask=None) -> None:
        super().__init__()
        key = jax.random.PRNGKey(42)
        self.layers = [eqx.nn.Linear(5*5,64,key=key,use_bias=True), 
                       jax.nn.relu,
                       eqx.nn.Linear(64,out_size,key=key,use_bias=True),
                      ]
        self.mask =  jnp.array(mask,dtype=bool)

    def __call__(self, obs,mask) -> Any:
        obs = obs.reshape(-1)
        for layer in self.layers:
            obs = layer(obs)
        # print('self mask',self.mask)
        # print("PG res mask",~self.mask,mask)

        # print("LM res mask",mask)  
        mask = ~self.mask | mask
              
        masked_logits = jnp.where(mask,-1e9,obs)
        # print(masked_logits)
        logits = jax.nn.softmax(masked_logits)
        # print(mask,)
        # input()
        # print('logits',logits)
        # input()
        return logits
    

    
class LinearModel(Model):
    layers:list
    mask: tuple
    def __init__(self,out_size=10,mask=None) -> None:
        super().__init__()
        key = jax.random.PRNGKey(42)
        self.layers = [
            eqx.nn.Linear(5*5,64,key=key,use_bias=True),
            jax.nn.relu,
            eqx.nn.Linear(64,out_size,key=key,use_bias=True),
            ]
        self.mask = jnp.array(mask,dtype=bool)
    def __call__(self, obs,mask) -> Any:
        obs = obs.reshape(-1)
        for layer in self.layers:
            obs = layer(obs)
        # print('self_mask',self.mask)
        mask = ~self.mask | mask
        # print("LM res mask",mask)
        masked_logits = jnp.where(mask, -1e9,obs)

        logits = jax.nn.softmax(masked_logits)
        # print('logits',logits)


        return logits
    

