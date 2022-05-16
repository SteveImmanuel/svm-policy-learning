import numpy as np
import math

# state xa ya za ha la wa cos sin hb wb hc wc
# action px py pz d1x d1y d1z d2x d2y d2z d3x d3y d3z f pre spt1 spt2 spt3
np.random.seed(2022)


def gen_random_state_vector():
    xa = np.random.rand() * 5
    ya = np.random.rand() * 5
    ha = np.random.randn() * 1
    la = np.random.randn() * 5
    wa = np.random.randn() * 3
    hb = ha - np.random.randn() * .05
    wb = wa - np.random.randn() * .05
    hc = wa - np.random.randn() * .05
    wc = wa - np.random.randn() * .05
    angle = np.random.rand() * math.pi * 2
    return [xa, ya, ha, la, wa, math.cos(angle), math.sin(angle), hb, wb, hc, wc]


def gen_random_action_vector():
    raise NotImplementedError()


if __name__ == '__main__':
    state_vect = gen_random_state_vector()
    print(state_vect)
    state_vect = gen_random_state_vector()
    print(state_vect)
    state_vect = gen_random_state_vector()
    print(state_vect)
    state_vect = gen_random_state_vector()
    print(state_vect)
