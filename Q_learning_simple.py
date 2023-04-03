#Q-learning example


# *____#
# *->current position, reach # to win
import numpy as np


table = np.array([0,0,0,0,0,100])

# 2 actions in each position
# first row -> move left
# second row -> move right
#Q_table=np.random.random((2,6))
Q_table=np.zeros((2,6))

# prob
ep=0.2

#return decay
gama=0.9

#learning rate
r=0.2

action=np.array([-1,1])

# [st,st+1,rt,at]
replay_buffer=[]

def reward(pos):
    return table[pos]


# epi-greedy policy
def choose_action(pos):
    p=np.random.uniform()
    # choose max Q action
    if p<=1-ep:
        act=np.argmax(Q_table[:, pos])
    else:
        p=np.array([0.5,0.5])
        act=np.random.choice(action,p=p)
    return action[act]


def param_upgrade():
    for i in replay_buffer:
        st, st1, rt, at=i
        qj=Q_table[at,st]
        qj1=np.max(Q_table[:,st1])
        yj=rt+gama*qj1
        Q_table[at,st] -= r*(qj-yj)


def draw(position):
    map=[]
    for i in range(6):
        if i!=position:
            map.append('_')
        elif i!=5:
            map.append('#')
        else:
            map.append('|')
    print(map)

def train():
    epoch=10
    for i in range(epoch):
        replay_buffer.clear()
        position = 0
        time=0
        while position != 5:
            if position==0:
                act=1
            else:
                act=choose_action(position)
            r=reward(position+act)
            replay_buffer.append([position,position+act,r,act])
            param_upgrade()
            position=position+act
            draw(position)
            time+=1
        print(time)


train()
print(Q_table)