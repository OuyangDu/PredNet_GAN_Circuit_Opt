import pandas as pd

class BO_Masker():
    def __init__(self, bo_info):
        self.bo_info = bo_info
        self._get_attribute()

    def _get_attribute(self):
        self.n_bo_units = {module: len(self.bo_info[module][self.bo_info[module]['is_bo'] == True]) for module in self.bo_info}
        self.n_non_bo_units = {module: len(self.bo_info[module][self.bo_info[module]['is_bo'] == False]) for module in self.bo_info}
        self.n_units = {module: len(self.bo_info[module]) for module in self.bo_info}

    def sample_bo_indices(self, module, n_units, is_bo=True, sample_method='random', put_channel_first=True):
        '''
        if number of units is larger than the units available, return all the indices. If is bo is None, random sample from all the units
        '''
        if not is_bo and sample_method == 'rank':
            raise ValueError('sample_method cannot be rank for non-bo')

        bo_module = self.bo_info[module]
        if is_bo is None:
            pass
        else:
            bo_module = bo_module[bo_module['is_bo'] == is_bo] # select bo or non-bo

        if n_units >= len(bo_module):
            print('n_units is larger than the available units, return all the indices')
            n_sample = len(bo_module)
        else:
            n_sample = n_units
        if sample_method == 'random':
            sampled_df = bo_module.sample(n=n_sample)
            neuron_id_list = sampled_df['neuron_id'].to_list()
            bav_list = sampled_df['bav'].to_list()
        elif sample_method == 'rank':
            ranked_df = bo_module.sort_values(by='bo_only_rank', ascending=True)
            sampled_df = ranked_df.head(n_sample)
            neuron_id_list = sampled_df['neuron_id'].to_list()
            bav_list = sampled_df['bav'].to_list()

        if put_channel_first:
            neuron_id_list = [(neuron_id[2], neuron_id[0], neuron_id[1]) for neuron_id in neuron_id_list]
        else: # channel last
            pass # the original data was orgonized in this way
        return neuron_id_list, bav_list

    def sample_bo_indices_modules(self, n_units_dict={}, is_bo=True, sample_method='random', put_channel_first=True):
        '''
        n_units: dict, the number of units to sample for each module
        '''
        sampled_neuron_id_list = {}
        sampled_bav_list = {}
        for module in n_units_dict:
            sampled_neuron_id_list[module], sampled_bav_list[module] = self.sample_bo_indices(module, n_units_dict[module], is_bo, sample_method, put_channel_first)

        r_mask_indices = [sampled_neuron_id_list.get('R{}'.format(i), []) for i in range(4)]
        e_mask_indices = [sampled_neuron_id_list.get('E{}'.format(i), []) for i in range(4)]
        return e_mask_indices, r_mask_indices, sampled_bav_list

