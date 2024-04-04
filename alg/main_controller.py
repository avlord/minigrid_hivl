from dataset import *
from models import *
from optimizer import *
from controller import *

import pickle

def train():

    skill_names = {  
                0:'Root',
                1:'Move',
                2:'Door'}
    
    max_skill_id = len(skill_names)
    inv_actions = {
                    0 + max_skill_id: 'Turn_Left', 
                    1 + max_skill_id: 'Turn_Right', 
                    2 + max_skill_id: 'Move',
                    3 + max_skill_id: 'Pickup', 
                    4 + max_skill_id: 'Drop', 
                    5 + max_skill_id: 'Use', 
                    # 6 + max_skill_id: 'Terminate'
                  }
    pickles = []
    for i in range(6):

        with open(f'data{i}.pickle', 'rb') as handle:
            pickle_file = pickle.load(handle)
            pickles.append(pickle_file)
        break
    dataset = FromPickle(pickles,max_skill_id)
    
    # zero_mask = (1,1,1,1,1,1,1,1,1,1)
    root_mask = (1,1,1,0,0,0,0,0,0,0)
    move_mask =(0,0,1,1,1,1,0,0,0,1)
    door_mask = (0,1,0,0,0,0,1,1,1,1)

    root = PGModel(out_size=10,mask=root_mask)
    A = LinearModel(out_size=10,mask=move_mask) #Move
    B = LinearModel(out_size=10,mask=door_mask) #Door
    
    root.add_model(A)
    root.add_model(B)

    optim_A, opt_state_A = init_optimizer(A,lr=1e-3)
    optim_B, opt_state_B = init_optimizer(B,lr=1e-3)

    optim_root, opt_state_root = init_optimizer(root,lr=1e-3)


    actions = inv_actions
    skills = {0:(root,(optim_root, opt_state_root)),1:(A,(optim_A,opt_state_A)),2:(B,(optim_B,opt_state_B))}
    controller = Controller(dataset,root=0,actions=actions,skills=skills,skill_names=skill_names)

    state = controller.reset()
    for epoch in range(200*6):
        print('Epoch',epoch)
        
        controller.act(state)
        state = controller.reset()
        for model_id in controller.model_datasets:
            for dp in controller.model_datasets[model_id]:
                # print(model_id)
                if dp.reward is None:
                    # if update:
                    x,mask,y = dp.x,dp.mask,dp.y
                    model,model_state = controller.skills[dp.model_id]
                    (loss_val,aux), grads = loss_fun(model,(x,mask),y)
                    # print('loss',loss_val)
                    model, model_state = update_model(model,grads,*model_state)
                    controller.skills[model_id] = (model,model_state)
                else:
                    # if model_id != 0:
                    #     continue
                    x,mask,y,reward = dp.x,dp.mask,dp.y,dp.reward
                    model,model_state = controller.skills[dp.model_id]
                    print('reward for model id', dp.model_id,reward)
                    # if type(reward)==int:
                    #     reward = reward[0]
                    reward = -(np.array(reward)-1).sum()
                    loss,grads=compute_loss_pi(model,mask,x,reward) #model,mask,obs,adv,idx
        #         # print('grads',loss)
                    model,model_state = update_model(model,grads,model_state[0],model_state[1])
                    controller.skills[model_id] = (model,model_state)
        #         self.skills[model_id] = (model,model_state)
        #         break
                    
        controller.model_datasets = defaultdict(list)


        # print(controller.model_datasets)
        # print('Epoch',epoch)
        # traj = dataset.sample()
        # for s_i, (s,a) in enumerate(traj):
        #     if s_i == 0:
        #         idx,pred = root.step(s)
        #         next_model = root.sub_models[idx]
        #         term = False
            
        #     if term:
        #         idx,pred = root.step(s)
        #         next_model = root.sub_models[idx]

        #     pred = jnp.argmax(next_model(s))
        #     term = True if pred == 6 else False
        #     (loss_val,aux), grads = loss_fun(next_model,s,a)
        #     next_model, optim_A, opt_state_A = update_model(next_model,grads,optim_A,opt_state_A)



        #     root.sub_models[idx] = next_model
        #     print(f'loss:{-(loss_val+1)}\n model:{model_map[int(idx)]},s_i:{s_i}')
        #     print('predicted_action',inv_actions[int(aux)],inv_actions[a])
        #     root_loss,grads_root=compute_loss_pi(root,s,-(loss_val-1),idx)
        #     root, optim_root, opt_state_root = update_model(root,grads_root,optim_root,opt_state_root)
    return root
train()



