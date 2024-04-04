import jax.numpy as jnp
import numpy as np

class Dataset():
    def __init__(self) -> None:
        s1 = (0.5,1,) # when we see 0, we do action 1
        s2 = (-0.5,-1) # when we see -1, we do action 2
        s3=(2,4)
        s4=(-2,-4)
        s5=(3,6)
        s6=(-3,-6)

        self.data = [[s1,s2,s3,s4,s5,s6],]
    def sample(self,):
        return self.data[0]    


class FromPickle(Dataset):
    def __init__(self,pickles,max_skill_id) -> None:
        super().__init__()
        # [0, 'left', 'Turn left'],
    #     [1, 'right', 'Turn right'],
    #     [2, 'forward', 'Move forward'],
    #     [3, 'pickup', 'Pick up an object'],
    #     [4, 'drop', 'Unused'],
    #     [5, 'toggle', 'Toggle/activate an object'],
    #     [6, 'done', 'Unused']
        self.max_skill_id=max_skill_id
        
        self.data = pickles
        self.actions = {
            'Turn_Left': 0,
            'Turn_Right': 1,
            'Move': 2,   
            'Pickup': 3,
            'Drop': 4,
            'Use': 5,
            'Terminate':6,
        }
        
        for s,a in zip(self.data[0]['s'],self.data[0]['a']):
            # print(dp[1])
            print('action:',a,'dir:',s[1])   
            print(s[0])
        
        self.data = [(self.data[i]['s'],self.data[i]['a']) for i in range(len(self.data))]
    
        
    def sample(self,):
        #carefull about direction
        j = np.random.randint(len(self.data))

        traj = [(jnp.array(s[0]),a+self.max_skill_id) for (s,a) in zip(self.data[j][0],self.data[j][1])]    
        traj[0]
        # print(traj)
        return traj

class DummyDataset(Dataset):
    def __init__(self,map_layout) -> None:
        super().__init__()
        self.map_layout = map_layout
        self.actions = {
            'Move': 0,   
            'Turn_Left': 1,
            'Turn_Right': 2,
            'Pickup': 3,
            'Use': 4,
            'Terminate':5,
        }
        ## S (x,y,alpha,key) ##
        s1 = ((1,1,0,0),'Move')  ### 0 angle ->, counter clock-wise
        s2 = ((2,1,0,0),'Move')
        s3 = ((3,1,0,0),'Pickup')
        s4 = ((3,1,0,1),'Turn_Left')
        s5 = ((3,1,1,1),'Turn_Left')
        s6 = ((3,1,2,1),'Move')
        s7 = ((2,1,2,1),'Turn_Right')
        s8 = ((2,1,1,1),'Move')
        s9 = ((2,2,1,1),'Use')
        s10 = ((2,2,1,0),'Move')
        s11 = ((2,3,1,0),'Turn_Left')
        s12 = ((2,3,2,0),'Move')
        s12 = ((1,3,2,0),'Terminate')
        self.data = [[s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12],]
    
    def sample(self,):
        traj = [(jnp.array(s),self.actions[a]) for (s,a) in self.data[0]]    
        # print(traj)
        return traj

#   01234
    ##### 4   
    #G**# 3
    ##D## 2
    #S*K# 1
    ##### 0    
#   01234   

# map_layout = [
#     "#####",
#     "#G**#",
#     "##D##",
#     "#S*K#",
#     "#####"
# ]

# dataset = DummyDataset(map_layout)
# sample = dataset.sample()
# print(sample)