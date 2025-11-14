import os
import numpy as np
from collections import defaultdict


class StateWithHandels(object):
    def __init__(self):
        self.state = None
        self.handels = None


class ExpStateProvider(object):
    def __init__(self, data_folder):
        self.states = defaultdict(StateWithHandels)

        package_directory = os.path.dirname(os.path.abspath(__file__))
        for file_name in os.listdir(os.path.join(package_directory, data_folder)):
            path = os.path.join(package_directory, data_folder, file_name)
            data = np.load(path)
            if file_name.endswith('_handles.npy'):
                name = file_name[0: -len('_handles.npy')]
                self.states[name].handels = data
            elif file_name.endswith('.npy'):
                name = file_name[0:-len('.npy')]
                self.states[name].state = data

    def get_state(self):
        name = np.random.choice(list(self.states.keys()))
        st = self.states[name]
        tot_intens = [np.sum(img) for img in st.state]
        return st.state, tot_intens, st.handels
