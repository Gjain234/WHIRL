import numpy as np
import pandas as pd
import pickle
import yaml
from pathlib import Path

from typing import Callable, Tuple

from Dataset import Dataset
from dataset_utils import extract_mdps, extract_budget, split_offline_traj
# from project_code.utils import get_from_command_line

class AAAI_22_Dataset(Dataset):
    # File containing high-level information about the dataset
    conf = yaml.safe_load(Path("aaai_22_info.yml").read_text())

    def __init__(self):
        # Get parameters from command line
        # args = get_from_command_line([
        #     "policy_type", "num_timesteps", "num_trajectories", "prior_coefficient"
        # ])

        # Save the policy type and number of timesteps
        self._policy_type: str = 'round_robin'
        self._num_states: int = AAAI_22_Dataset.conf["num_states"]
        self._num_actions: str = AAAI_22_Dataset.conf["num_actions"]
        self._starting_state_probs = np.array(AAAI_22_Dataset.conf["starting_state_probs"], dtype='float64')

        # Get offline trajectory
        self._num_timesteps = 11
        _features, _offline_traj, self._policy,self._user_ids = self._get_offline_traj(self._policy_type, self._num_timesteps)

        # Break trajectories by beneficiaries into num_trajectories
        self._num_trajectories = 1
        split_idxs = split_offline_traj(_offline_traj, self._num_trajectories)
        self._features = np.array(_features[split_idxs], dtype='float64')
        self._offline_traj = np.array(_offline_traj[split_idxs], dtype='int64')
        self._num_beneficiaries = self._offline_traj.shape[1]

        # Get the effective budget for each trajectory
        self._budget = extract_budget(self.get_offline_trajectories())  # type: ignore

        # Extract the MDPs from the trajectories
        self._prior_coefficient = 1
        # self._mdp = extract_mdps(_offline_traj, self._prior_coefficient)

    @property
    def num_features(self) -> int:
        return self._features.shape[-1]

    @property
    def num_states(self) -> int:
        return self._num_states

    @property
    def num_actions(self) -> int:
        return self._num_actions

    @property
    def num_trajectories(self) -> int:
        return self._num_trajectories
    
    @property
    def num_beneficiaries(self) -> int:
        return self._num_beneficiaries

    @property
    def num_timesteps(self) -> int:
        return self._num_timesteps

    @property
    def budget(self) -> int:
        return self._budget

    @property
    def starting_state_probs(self) -> np.ndarray:
        return np.ones(self.num_states) / self.num_states
    
    def get_start_state_prob(self) -> np.ndarray:
        return self._starting_state_probs

    def get_features(self) -> np.ndarray:
        return self._features
    
    def get_offline_trajectories(self) -> np.ndarray:
        return self._offline_traj
    
    def get_mdps(self) -> np.ndarray:
        return self._mdp

    # Helper function to get offline trajectories from data dump
    def _get_offline_traj(self, policy_type: str, num_timesteps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
        # LOAD DATA
        # Load dataframes from offline data
        intervention_df = pd.read_csv(AAAI_22_Dataset.conf["intervention_df_path"])
        analysis_df = pd.read_csv(AAAI_22_Dataset.conf["analysis_df_path"])
        #   TODO: Get rid of this hack. Save the pilot data in a separate csv file
        with open(AAAI_22_Dataset.conf["features_dump_path"], "rb") as fr:
            pilot_user_ids, pilot_static_features = pickle.load(fr)

        # PRE-PROCESS DATA
        # If policy_type option is rr or random, we will fetch real world round robin trajectories
        policy_equiv_dict = {"rr": "round_robin", "random": "round_robin"}

        # filter only entries for given policy_type
        policy_analysis_df = analysis_df.loc[
            analysis_df["arm"] == policy_type
        ]

        # SUMMARIZE DATA
        # note all the state columns
        state_cols = [f"week{i}_state" for i in range(num_timesteps + 1)]
        # state 0 : sleeping engaging, 1: sleeping non-engaging, 6: engaging, 7: sleeping non engaging
        # to convert this into 0: non-engaging, 1: engaging
        state_df = (policy_analysis_df[state_cols] % 2 == 0).astype(int)
        state_matrix = state_df.values
        assert state_matrix.shape[1] == num_timesteps + 1
        clusters = policy_analysis_df["cluster"]
        # Reward is same as states, but exists only for num_timesteps-1 steps
        reward_matrix = np.copy(state_matrix[:, :-1])

        # filter intervention df for given policy_type
        policy_intervention_df = intervention_df[
            intervention_df["exp_group"] == policy_type
        ]
        
        
        # note columns for every week
        week_cols = [f"week{i+1}" for i in range(num_timesteps)]
        # Convert weeks from long to wide, and mark 1 for every week when intervened
        policy_intervention_df = (
            (
                policy_intervention_df.pivot(
                    index="user_id", columns="intervene_week", values="intervene_week"
                )[week_cols].isna()
                != 1
            )
            .astype(int)
            .reset_index()
        )
        # merge policy_intervention_df with policy_analysis_df, to get interventions actions
        # in the same order as state matrix
        actions_df = pd.merge(
            policy_analysis_df[["user_id"]],
            policy_intervention_df[["user_id"] + week_cols],
            how="left",
        ).fillna(0)
        action_matrix = actions_df[week_cols].values
        assert action_matrix.shape[0] == state_matrix.shape[0]
        assert action_matrix.shape[1] == num_timesteps

        # Convert to offline trajectory format
        offline_trajectory = np.stack([np.copy(state_matrix[:, :-1]), action_matrix, np.copy(state_matrix[:, 1:]), reward_matrix], axis=-1)
        assert offline_trajectory.shape[0] == state_matrix.shape[0] == action_matrix.shape[0] == reward_matrix.shape[0]
        assert offline_trajectory.shape[1] == num_timesteps == action_matrix.shape[1] == reward_matrix.shape[1] == state_matrix.shape[1] - 1
        assert offline_trajectory.shape[2] == 4

        # GET FEATURES
        # Get features for training the predictive model
        feature_df = pd.DataFrame(pilot_static_features)
        feature_df["user_id"] = pilot_user_ids
        feature_df = feature_df.set_index("user_id")
        policy_feature_df = feature_df.loc[policy_analysis_df["user_id"]]
        policy_features = policy_feature_df.values

        # DEFINE POLICY
        # Define the policy used to generate the offline trajectory (for OPE)
        # TODO: Implement this
        policy = None
        return policy_features, offline_trajectory, clusters.values, pilot_user_ids


# Run unit tests if run as a script
if __name__ == "__main__":
    dataset = AAAI_22_Dataset()
    L = 11
    traj_rmab = dataset._get_offline_traj('rmab', L)
    mdps = extract_mdps(np.array(traj_rmab[1], dtype='int64')[None], 1)