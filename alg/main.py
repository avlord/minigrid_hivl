from dataset import *
from models import *
from optimizer import *

def train():
    map_layout = [
    "#####",
    "#G**#",
    "##D##",
    "#S*K#",
    "#####"
    ]

    actions = {
            'Move': 0,   
            'Turn_Left': 1,
            'Turn_Right': 2,
            'Pickup': 3,
            'Use': 4,
            'Terminate':5,
        }

    dataset = DummyDataset(map_layout)
    # sample = dataset.sample()

    root = PGModel()
    A = LinearModel(mask=(1,1,1,0,0,0)) #Move
    B = LinearModel(mask=(0,0,0,1,1,1)) #Door

    model_map = {0:'Move',1:'Door'}

    root.add_model(A)
    root.add_model(B)
    optim_A, opt_state_A = init_optimizer(A,lr=1e-1)

    optim_root, opt_state_root = init_optimizer(root,lr=1e-2)

    for epoch in range(100):
        print('Epoch',epoch)
        traj = dataset.sample()
        
        for s_i, (s,a) in enumerate(traj):
            idx,pred = root.step(s)
            # idx = 0
            # print('pred',s, pred,idx)
            next_model = root.sub_models[idx]
            (loss_val,aux), grads = loss_fun(next_model,s,a)
            
            next_model, optim_A, opt_state_A = update_model(next_model,grads,optim_A,opt_state_A)
            root.sub_models[idx] = next_model

            
            # print(loss_val,s,(a,aux))
            # root.add_training_example((s,a,idx,loss_val))
            print(f'loss:{-loss_val}\n model:{model_map[int(idx)]},s_i:{s_i}')
            root_loss,grads_root=compute_loss_pi(root,s,-loss_val,idx)
            # print('loss',root_loss)

            root, optim_root, opt_state_root = update_model(root,grads_root,optim_root,opt_state_root)


    return root

train()



