import gymnasium 
import numpy as np
import pickle
from dataset import FromPickle
from minigrid.wrappers import SymbolicObsWrapper, FullyObsWrapper
np.random.seed(42)

env = FullyObsWrapper(gymnasium.make("MiniGrid-DoorKey-5x5-v0",render_mode='human'))
print(env.action_space)
# print(env.action_space.sample())
# [0, 'left', 'Turn left'],
#     [1, 'right', 'Turn right'],
#     [2, 'forward', 'Move forward'],
#     [3, 'pickup', 'Pick up an object'],
#     [4, 'drop', 'Unused'],
#     [5, 'toggle', 'Toggle/activate an object'],
#     [6, 'done', 'Unused']

obs = env.reset()
obs =obs[0]
img, direction,mission = obs['image'], obs['direction'], obs['mission']

dataset = {'s':[],'a':[]}
while True:
    env.render()
    a = int(input('input action'))
    dataset['s'].append((img[:,:,0]+img[:,:,2],direction))
    dataset['a'].append(a)
    obs,reward,done,truncated,info = env.step(a)
    img, direction,mission = obs['image'], obs['direction'], obs['mission']
    # print('shape',img.shape)
    
    # print(dataset)
    # break
    # print(done,truncated)
    if done or truncated:
        with open('data0.pickle', 'wb') as handle:
            pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # break
        pickles = []
        with open(f'data0.pickle', 'rb') as handle:
                pickle_file = pickle.load(handle)
                pickles.append(pickle_file)
            
        dataset = FromPickle(pickles,3)
        break