#Q-learning example


# *____#
# *->current position, reach # to win
import numpy as np


table = np.array([0,0,-2,10,-2,20])

# 2 actions in each position
# first row -> move left
# second row -> move right
Q_table=np.zeros((3,6))

# prob
ep=0.2

#return decay
gama=0.3

#learning rate
r=0.01

action=np.array([-1,1,0])

# [st,st+1,rt,at]
replay_buffer=[]

def reward(pos):
    return table[pos]


# epi-greedy policy
def choose_action(pos):
    p=np.random.uniform()
    if pos==0:
        p = np.array([0, 0.5, 0.5])
        act = np.random.choice(np.arange(0,3), p=p)
        return act
    elif pos==5:
        p = np.array([0.5, 0, 0.5])
        act = np.random.choice(np.arange(0,3), p=p)
        return act
    # choose max Q action
    if p<=1-ep:
        act=np.argmax(Q_table[:, pos])
    else:
        p=np.array([0.33,0.33,0.34])
        act=np.random.choice(np.arange(0,3),p=p)
    return act


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
    epoch=100
    for i in range(epoch):
        replay_buffer.clear()
        position = 0
        time=0
        while time<50:
            act=choose_action(position)
            r=reward(position+action[act])
            replay_buffer.append([position,position+action[act],r,act])
            param_upgrade()
            position=position+action[act]
            draw(position)
            time+=1
        print(time)


train()
print(Q_table)