import os
import pickle
import numpy as np


class DataLoader(object):
    def __init__(self, path, num_hooks=2, shape='C',
                 bad_grasps=False,
                 start_dir_ind=0, start_file_ind=0,
                 end_dir_ind=5, end_file_ind=4, num_dp_per_file=10000):
        self.num_hooks = num_hooks
        assert self.num_hooks in [2, 3, 4]
        self.root_path = path
        self.path = self.root_path + '/' + str(self.num_hooks) + '_hooks'
        self.hook_shape = shape
        if self.hook_shape != 'C':
            self.path = self.path + '_' + self.hook_shape
        self.bad_grasps = bad_grasps
        self.end_dir_ind = end_dir_ind
        self.end_file_ind = end_file_ind
        self.start_dir_ind = start_dir_ind
        self.current_dir_ind = start_dir_ind
        self.start_file_ind = start_file_ind
        self.current_file_ind = start_file_ind
        self.current_in_file_data_point_ind = 0
        self.num_dp_per_file = num_dp_per_file
        self.num_passed_data_point = 0
        self.data = dict()
        self.init()
        self.finished = False

    @property
    def current_dir(self):
        return self.path + '/dir_' + str(self.current_dir_ind)

    def init(self):
        self.num_passed_data_point = 0
        self.load_next_file(init=True)

    def load_next_file(self, init=False):
        if init:
            self.current_dir_ind = self.start_dir_ind
            self.current_file_ind = self.start_file_ind
        else:
            self.current_file_ind += 1
            if self.current_file_ind > self.end_file_ind:
                self.current_dir_ind += 1
                self.current_file_ind = 0

        if self.current_dir_ind > self.end_dir_ind:
            self.current_dir_ind = self.start_dir_ind
            self.current_file_ind = self.start_file_ind

        self.load_file(self.current_dir_ind, self.current_file_ind)
        self.current_in_file_data_point_ind = 0

    def load_file(self, dir_ind, file_ind):
        # print(os.path.join(self.path, 'dir_' + str(dir_ind), 'good_grasps_' + str(file_ind) + '.npy'))
        good_grasps = np.load(os.path.join(self.path, 'dir_' + str(dir_ind), 'good_grasps_' + str(file_ind) + '.npy'))
        good_grasps_features = np.load(
            os.path.join(self.path, 'dir_' + str(dir_ind), 'good_grasp_features_' + str(file_ind) + '.npy'))
        with open(os.path.join(self.path, 'dir_' + str(dir_ind), 'good_grasp_mjc_states_' + str(file_ind) + '.mjc'),
                  'rb') as f:
            good_grasps_mjc_states = pickle.load(f)
        self.data = {
            'good_grasps': good_grasps,
            'good_grasp_features': good_grasps_features,
            'good_grasp_mjc_states': good_grasps_mjc_states,
        }
        if self.bad_grasps:
            bad_grasps = np.load(
                os.path.join(self.path, 'dir_' + str(dir_ind), 'bad_grasps_' + str(file_ind) + '.npy'))
            bad_grasps_features = np.load(
                os.path.join(self.path, 'dir_' + str(dir_ind), 'bad_grasp_features_' + str(file_ind) + '.npy'))
            with open(os.path.join(self.path, 'dir_' + str(dir_ind), 'bad_grasp_mjc_states_' + str(file_ind) + '.mjc'),
                      'rb') as f:
                bad_grasps_mjc_states = pickle.load(f)
            self.data.update({
                'bad_grasps': bad_grasps,
                'bad_grasp_features': bad_grasps_features,
                'bad_grasp_mjc_states': bad_grasps_mjc_states,
            })

    def get_next_good_state(self):
        grasp = self.data['good_grasps'][self.current_in_file_data_point_ind]
        feature = self.data['good_grasp_features'][self.current_in_file_data_point_ind]
        mjc_state = self.data['good_grasp_mjc_states'][self.current_in_file_data_point_ind]

        self.current_in_file_data_point_ind += 1
        if self.current_in_file_data_point_ind == self.num_dp_per_file:
            self.load_next_file()
        self.num_passed_data_point += 1

        return grasp, feature, mjc_state

    def get_random_batch(self, batch_size=256, good_data_portion=0.5):
        num_good_data_point = int(batch_size * good_data_portion)
        g_inds = np.random.choice(a=np.arange(self.num_dp_per_file),
                                  size=num_good_data_point, replace=False)
        g_grasps = self.data['good_grasps'][g_inds]
        g_features = self.data['good_grasp_features'][g_inds]
        g_labels = np.ones(shape=(num_good_data_point,))

        b_inds = np.random.choice(a=np.arange(self.num_dp_per_file),
                                  size=int(batch_size - num_good_data_point), replace=False)
        b_grasps = self.data['bad_grasps'][b_inds]
        b_features = self.data['bad_grasp_features'][b_inds]
        b_labels = np.zeros(shape=(batch_size - num_good_data_point,))

        grasps = np.concatenate((g_grasps, b_grasps), axis=0)
        features = np.concatenate((g_features, b_features), axis=0)
        labels = np.concatenate((g_labels, b_labels))
        return grasps, features, labels

    def get_good_random_batch(self, batch_size=256):
        g_inds = np.random.choice(a=np.arange(self.num_dp_per_file), size=batch_size, replace=False)
        g_grasps = self.data['good_grasps'][g_inds]
        g_features = self.data['good_grasp_features'][g_inds]
        return g_grasps, g_features

    def get_bad_random_batch(self, batch_size=256):
        b_inds = np.random.choice(a=np.arange(self.num_dp_per_file), size=batch_size, replace=False)
        b_grasps = self.data['bad_grasps'][b_inds]
        b_features = self.data['bad_grasp_features'][b_inds]
        return b_grasps, b_features


script_path = os.path.dirname(__file__)
sg_data_path = os.path.join(script_path, 'SG_data')
data_loader = DataLoader(path=sg_data_path,
                         shape='C+',
                         start_dir_ind=0,
                         end_dir_ind=4,
                         end_file_ind=0,
                         num_dp_per_file=10000,
                         num_hooks=3)

done = False
while not done:
    grasp, feature, mjc_state = data_loader.get_next_good_state()

# Mujoco and the environment based on it are required to visualise the recorded mjc state.
# Please refer to https://github.com/IanYangChina/GAM-paper-codes for more details.