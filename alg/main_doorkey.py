from dataset import *
from models import *
from optimizer import *
import pickle

def train():
    inv_actions = {
                    0: 'Turn_Left', 
                    1: 'Turn_Right', 
                    2: 'Move',
                    3: 'Pickup', 
                    4: 'Drop', 
                    5: 'Use', 
                    6: 'Terminate'
                  }
    model_map = { 0:'Move',
                1:'Door'}
    pickles = []
    for i in range(6):
        with open(f'data{i}.pickle', 'rb') as handle:
            pickle_file = pickle.load(handle)
            pickles.append(pickle_file)
    
    dataset = FromPickle(pickles)
    
    root = PGModel()
    A = LinearModel(mask=(1,1,1,0,0,0,1)) #Move
    B = LinearModel(mask=(0,0,0,1,1,1,1)) #Door
    
    root.add_model(A)
    root.add_model(B)
    optim_A, opt_state_A = init_optimizer(A,lr=1e-3)
    optim_root, opt_state_root = init_optimizer(root,lr=1e-3)
    for epoch in range(100):
        print('Epoch',epoch)
        traj = dataset.sample()
        for s_i, (s,a) in enumerate(traj):
            if s_i == 0:
                idx,pred = root.step(s)
                next_model = root.sub_models[idx]
                term = False
            
            if term:
                idx,pred = root.step(s)
                next_model = root.sub_models[idx]

            pred = jnp.argmax(next_model(s))
            term = True if pred == 6 else False
            (loss_val,aux), grads = loss_fun(next_model,s,a)
            next_model, optim_A, opt_state_A = update_model(next_model,grads,optim_A,opt_state_A)



            root.sub_models[idx] = next_model
            print(f'loss:{-(loss_val+1)}\n model:{model_map[int(idx)]},s_i:{s_i}')
            print('predicted_action',inv_actions[int(aux)],inv_actions[a])
            root_loss,grads_root=compute_loss_pi(root,s,-(loss_val-1),idx)
            root, optim_root, opt_state_root = update_model(root,grads_root,optim_root,opt_state_root)
    return root
train()



