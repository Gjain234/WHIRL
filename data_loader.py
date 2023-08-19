import os
import pickle

import numpy as np
import pandas as pd

from CONFIG import BASE_PATHS, CACHE_FOLDER_PATH, DATA_FILE_PATHS, FEAT_NAMES, STUDY_DATES, INTERV_ID_DICT, INTERV_SCHED_INFO
from fast_state_action_computation import compute_states, compute_actions
from fast_listenership_metrics_computation import compute_metrics
from training.data import load_data as legacy_load_data
from training.dataset import _preprocess_call_data as _legacy_preprocess_call_data
from training.dataset import preprocess_and_make_dataset as legacy_preprocess_and_make_dataset

from training_new.data import load_data
from training_new.dataset import _preprocess_call_data, preprocess_and_make_dataset


class StudyDataLoader():
    '''
    Class for loading data from studies after Nov 2021. 
    This specifically includes the Dec 2021, Jan 2021, March 2022, May 2022, July 2022 Studies
    '''
    def __init__(self, study_name, last_data_date, metrics_list = [], user_working_set = None, use_cache = True):
        self.study_name = study_name
        self.last_data_date = last_data_date
        self.use_cache = False
        beneficiary_data, call_data, experiment_df = self.load_preprocessed_data()
        if user_working_set is None:
            user_working_set = experiment_df.user_id.values
        self.user_working_set = user_working_set
        self.experiment_df = experiment_df
        call_data = call_data[call_data.user_id.isin(user_working_set)]
        beneficiary_data = beneficiary_data[beneficiary_data.user_id.isin(user_working_set)]

        feat_df, feat_info_dict = self.get_registration_features(beneficiary_data, call_data)
        state_df, state_info_dict = self.get_states(call_data)
        action_df, action_info_dict = self.get_actions(state_df)
        sched_action_df, sched_action_info_dict = self.get_sched_actions(state_df)
        metrics_df, metrics_info_dict = self.get_metrics(call_data, metrics_list)
        self.feat_df = feat_df
        self.state_df = state_df
        self.action_df = action_df
        self.sched_action_df = sched_action_df
        self.metrics_df = metrics_df
        self.interv_start_week = self.get_week_indicators()
    
    def get_week_indicators(self):
        # TODO: Add registration end week indicator and all beneficiaires active week indicator
        week0_end = STUDY_DATES[self.study_name]['week0_end']
        intervention_start_week_end = STUDY_DATES[self.study_name]['intervention_start_week_end']
        intervention_start_week_num = (pd.to_datetime(intervention_start_week_end) - pd.to_datetime(week0_end)).days//7 + 1
        return intervention_start_week_num
        
    def get_states(self, call_data):
        state_df, state_info_dict = compute_states(call_data, self.study_name, self.last_data_date)
        return state_df, state_info_dict
    
    def get_registration_features(self, beneficiary_data, call_data):
        fname = f'{CACHE_FOLDER_PATH}/{self.study_name}_registration_feats.pickle'
        if os.path.exists(fname) and self.use_cache:
            print('Loading registration feats from saved file')
            with open(fname, 'rb') as file:
                feat_df, feat_info_dict = pickle.load(file)
        else:
            if self.study_name=='April21':
                features_dataset = legacy_preprocess_and_make_dataset(beneficiary_data, call_data)
            else:
                features_dataset = preprocess_and_make_dataset(beneficiary_data, call_data)
            
            user_ids, _, static_xs, _, _ = features_dataset

            mismatches = (~np.isin(self.user_working_set, user_ids)).sum()
            assert mismatches==0, f'mismatches: {mismatches}, total users in Exp: {len(self.user_working_set)}, users in feat: {len(user_ids)}'
            static_xs = static_xs.astype(np.float32)

            static_features = np.array(static_xs, dtype=float)
            static_features = static_features[:, : -8]

            feat_df = pd.DataFrame(static_features, columns = FEAT_NAMES)

            feat_info_dict = {'full_features_dataset': features_dataset,
                                'user_ids': user_ids}
            
            with open(fname, 'wb') as file:
                pickle.dump([feat_df, feat_info_dict], file)

        return feat_df, feat_info_dict

    def get_actions(self, state_df):
        interv_data = pd.read_csv(f'data/{BASE_PATHS[self.study_name]}/interventions_data.csv')
        interv_ids = list(INTERV_ID_DICT[self.study_name].values())
        interv_data = interv_data[interv_data.beneficiary_id.isin(self.user_working_set) &
                                interv_data.intervention_id.isin(interv_ids) &
                                (pd.to_datetime(interv_data.intervention_date)<=self.last_data_date)]
        action_df, action_info_dict = compute_actions(interv_data, self.study_name, state_df)
        
        return action_df, action_info_dict
    
    def get_sched_actions(self, state_df):
        base_path = f'data/{BASE_PATHS[self.study_name]}/{INTERV_SCHED_INFO[self.study_name]["path"]}'
        n_interv = INTERV_SCHED_INFO[self.study_name]['n_interv']
        interv_start_date = STUDY_DATES[self.study_name]
        interv_file_pattern = INTERV_SCHED_INFO[self.study_name]['fname'] 
        # interv_dates = [pd.to_datetime([interv_start_date])+pd.to_timedelta(int(i*7), 'd') 
        #                                 for i in range(n_interv)]
        interv_file_names = [interv_file_pattern.format(i) for i in range(1, n_interv+1)]
        interv_sched_data = pd.DataFrame({
            'beneficiary_id': [],
            'intervention_date': [],
            'intervention_success': []
        })
        # TODO: interv date is already present in the intervention csv files
        # for interv_date, interv_file_name in zip(interv_dates, interv_file_names):
        for interv_file_name in interv_file_names:
            interv_df = pd.read_csv(f'{base_path}/{interv_file_name}')[['user_id', 'date']].rename(columns={
                'user_id': 'beneficiary_id',
                'date': 'intervention_date'
            })
            interv_df['intervention_success'] = 1
            interv_date = interv_df['intervention_date'].iloc[0]
            if interv_date<self.last_data_date:
                interv_sched_data = pd.concat([interv_sched_data, interv_df])
            else:
                break
        interv_sched_data.index = np.arange(len(interv_sched_data))

        action_df, action_info_dict = compute_actions(interv_sched_data, self.study_name, state_df)
        
        return action_df, action_info_dict


    def get_metrics(self, call_data, metrics_list):
        metrics_df, metrics_info_dict = compute_metrics(call_data, metrics_list, self.study_name, self.last_data_date)
        return metrics_df, metrics_info_dict

    def load_preprocessed_data(self):
        ## Use the preprocessing functions from training directory 
        ## to get call and registration feature data
        ## Use caching

        fname = f'{CACHE_FOLDER_PATH}/{self.study_name}_preprocessed_data.pickle'
        if os.path.exists(fname) and self.use_cache:
                print('Loading preprocessed call and beneficiary data from saved file')
                with open(fname, 'rb') as file:
                    beneficiary_data, call_data = pickle.load(file)
        else:
            local_config = {}
            local_config['read_sql'] = 0
            local_config['pilot_data'] = f'data/{BASE_PATHS[self.study_name]}'
            if self.study_name=='April21':
                beneficiary_data, call_data = legacy_load_data(local_config['pilot_data'])
                call_data = _legacy_preprocess_call_data(call_data)
            else:
                beneficiary_data, call_data = load_data(local_config)
                call_data = _preprocess_call_data(call_data)
            if self.use_cache:
                with open(fname, 'wb') as file:
                    pickle.dump([beneficiary_data, call_data], file)

        experiment_df = pd.read_csv(f"data/{BASE_PATHS[self.study_name]}/{DATA_FILE_PATHS['experiment']}")
        return beneficiary_data, call_data, experiment_df

if __name__ == "__main__":
    dl = StudyDataLoader('July22', '2022-9-07')

class DeploymentDataLoader():
    '''
    Class for loading data from deployment. Since deployment data dump isn't already available,
    this class will fetch the data directly from GCP SQL instance.
    '''
    def __init__(self, deployment_config):
        pass



class LegacyStudyDataLoader():
    '''
    Class for loading data from studies before Nov 2021. 
    This specifically includes the April 2021 study published in AAAI 2022
    '''
    def __init__(self, study_name):
        pass