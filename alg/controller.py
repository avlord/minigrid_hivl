import numpy as np
import jax.numpy as jnp
from optimizer import *
from collections import defaultdict

from dataclasses import dataclass

@dataclass
class DataPoint():
    model_id: int
    data_type: str
    x: np.array
    mask: np.array
    y: np.array
    logits: np.array
    reward: np.array

class Controller():
    def __init__(self,
                 dataset:list,
                 root,
                 actions,
                 skills,
                 skill_names,
                 ) -> None:
        self.stack = [root,]
        self.rewards = []
        self.dataset = dataset
        self.trajectory = None
        self.actions = actions #actions
        self.skills = skills #{'id':(model,model_state)}
        self.mask_len = len(self.actions) + len(self.skills) + 1 #terminate
        self.terminate = self.mask_len - 1
        self.root_id = root
        self.no_skills = False
        self.skill_names = skill_names
        self.state_id = 0
        self.cum_loss = 0
        self.queue = []

        self.model_datasets = defaultdict(list)
        # self.terminate = self.actions[len(self.actions)-1]
        print('actions',self.actions)
        print('skills', self.skill_names)
        print('terminate',self.terminate)

    def act(self, state:tuple[np.array,np.array]) -> None:
        counter = 0
        args = None
        print(self.stack)

        if len(self.stack) == 0:
            return
        
        i = 0
        while True:
            # i+= 1
            # if i > 10:
            #     return
            if state is None:
                return None, None

            model_id = self.stack[-1]
            model = self.get_model(model_id)
            mask = self.set_mask(model_id,args,counter)
            x, action = state[0], state[1]
            logits = model(x,mask,) ### door_id ###
            rnd = np.random.rand(1)
            # if rnd>0.05:
            p = np.array(logits)
            p /= p.sum()
                # print(p)
            pred = np.random.choice(np.arange(len(logits)),p=p)
            # pred = int(jnp.argmax(logits))
            # else:
                # pred = int(jnp.argmax(logits))
         

            if pred == self.terminate:
                print("predicted terminate",pred)
                self.add_to_training(model_id,
                                     data_type='terminate',
                                     x=x,
                                     mask=mask,
                                     y=action,
                                     logits=pred,
                                     reward=None)
                self.stack.pop()
                return state, model_id


            if pred in self.actions: 
                print("predicted", pred,self.actions[pred], self.actions[action],action)
                loss = self.get_loss(model,
                                     model_id,(state,mask),update=False)
                # loss  = 1 if pred == action else -1 
                self.cum_loss += loss
                # print("Added", loss)
                self.add_to_training(model_id,
                                     data_type='action',
                                     x=x,
                                     mask=mask,
                                     y=action,
                                     logits=logits,
                                     reward=None,
                                     loss=loss)

                state = self.get_new_state()

                
            elif pred in self.skill_names:
                print("predicted Skill", self.skill_names[pred], self.actions[action])
                self.add_to_training(model_id,
                                     data_type='skill',
                                     x=x,
                                     mask=mask,
                                     y=action,
                                     logits=pred,
                                     reward=None)

                self.stack.append(pred)
                state, args = self.act(state)
                # state = self.get_new_state()
                if state is None:
                    return (None, None)
                # break
            
            counter += 1
        
    def add_to_training(self,model_id,data_type,x,mask,y,logits,reward,loss=None):
        data = DataPoint(model_id=model_id,data_type=type,x=x,mask=mask,y=y,logits=logits,reward=reward)

        if data_type == 'action':
            self.model_datasets[model_id].append(data)
            
            # while len(self.queue) > 0 :
                # dp = self.queue.pop()
            for i in range(len(self.queue[:-1])):
                # print('Adding reward for model', model_id,loss, self.queue)
                if self.queue[i].reward is None:
                    self.queue[i].reward = [loss,]
                else:
                    self.queue[i].reward.append(loss)
            
                # self.model_datasets[model_id].append(self.queue[i])
                
                
        elif data_type == 'skill':
            self.queue.append(data)
        
        elif data_type == 'terminate':
            
            data.reward = self.cum_loss
            # print('terminate',self.cum_loss)
            # input()
            self.model_datasets[model_id].append(data)
            self.cum_loss = 0



    def set_mask(self,model_id,args,counter):
        mask = np.zeros(self.mask_len,dtype=bool)
        mask[0] = 1
        mask[model_id] = 1

        if len(self.stack)>1:
            ###Do not allow skills
            for key in self.skill_names:
                mask[key] = 1
        if counter == 0:
            ###Do not allow terminate:
            mask[self.terminate] = 1
        if args is not None:
            mask[args] = 1
        
        if self.no_skills:
            for key in self.skills:
                mask[key] = 1

        return mask
    
    def get_loss(self,model,model_id,state,update=True):
        (x,a),mask = state

        (loss_val,aux), grads = loss_fun(model,(x,mask),a)
        # if update:
        #     # (loss_val,aux), grads = loss_fun(next_model,s,a)
        #     # next_model, optim_A, opt_state_A = update_model(next_model,grads,optim_A,opt_state_A)
        #     model_state = self.skills[model_id][1]
        #     model,model_state = update_model(model,grads,*model_state)
        #     self.skills[model_id] = (model,model_state)
        #     ###UPDATE PG###
        #     self.update(loss_val,mask,x,a)
        return loss_val

    # def update(self,loss_val,mask,x,a):
    #     #pg update#
    #     for model_id in self.stack[:-1]:
    #         print('model_id',model_id,'reward',-(loss_val-1))
    #         model,model_state = self.skills[model_id]
    #         mask = self.set_mask(model_id)
    #         loss,grads=compute_loss_pi(model,mask,x,-(loss_val-1)) #model,mask,obs,adv,idx
    #         # print('grads',loss)
    #         model,model_state = update_model(model,grads,model_state[0],model_state[1])
    #         self.skills[model_id] = (model,model_state)
    #         break
            

    def get_new_state(self,):
        self.state_id += 1
        if self.state_id > len(self.trajectory) - 1:
            return
        state = self.trajectory[self.state_id]
        self.current_state = (state[0],state[1])
        return state[0],state[1]
    def get_model(self,model_id):
        return self.skills[model_id][0]
   
    def reset(self,):
        # self.update()
        self.trajectory = self.dataset.sample()
        if len(self.queue)>0:
            print('Reset')
            while self.queue:
                dp = self.queue.pop()
                self.model_datasets[dp.model_id].append(dp)
        self.cum_loss = 0
        self.stack = [self.root_id]
        self.state_id = 0
        return self.trajectory[0]

