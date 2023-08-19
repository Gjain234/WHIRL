import numpy as np
from sklearn.cluster import KMeans
from Dataset import Dataset


# Convert offline trajectory into a series of transition matrices
# TODO: Change the basis of the transition probabilities
# TODO(?): Implement clustered prior
def extract_mdps(
    trajs: np.ndarray,
    prior_coefficient: float,
    transition_type: str,
):
    """
    Given a set of RMAB trajectories, convert them to MDPs for each beneficiary.

    Args:
        trajs: A tf.Tensor of shape (num_trajectories, num_beneficiaries, num_timesteps, {s,a,s',r}) containing the trajectories.
        prior_coefficient: A float between 0 and inf. This coefficient controls how much weight to give to population level prior vs. empirical counts. A value of 0 means that the prior is ignored, and a value of inf means that the prior is used exclusively.
    
    Returns:
        prob_matrices: A tf.Tensor of shape (n_trajectories, n_beneficiaries, n_states, n_actions, n_states) containing the transition matrices for each beneficiary.
    """
    # Get a count of the number of s,a,s' transitions for each beneficiary
    count_matrices = get_count_matrices(trajs).astype(float)

    # Create priors at a population level prior
    pop_level_count = count_matrices.sum((0, 1))
    assert pop_level_count.min() > 0, "Some (s,a,s') tuples have no transitions at a population level. This is not allowed."
    pop_level_count /= pop_level_count.sum(-1, keepdims=True) # Normalize to get probabilities
  
    # Create a posterior at a trajectory level by combining the population level prior with the trajectory level counts
    if transition_type == 'population':
        # Create a posterior at a trajectory level by combining the population level prior with the trajectory level counts
        traj_level_count = count_matrices.sum(1)
        traj_level_count += prior_coefficient * pop_level_count[None, ...]
        traj_level_count /= traj_level_count.sum(-1, keepdims=True) # Normalize to get probabilities
        # Create a posterior at a beneficiary level by combining the trajectory level prior with the individual beneficiary counts
        count_matrices += prior_coefficient * traj_level_count[:, None, ...]
        count_matrices /= count_matrices.sum(-1, keepdims=True) # Normalize to get probabilities
    if transition_type == 'clusters':
        # Create a posterior at the cluster level by combining the population level prior with the trajectory level counts
        #   Get the number of clusters from the command line arguments
        clusters = 40
        #   Cluster the trajectories based on passive transition probabilities
        passive_counts = count_matrices[..., :, 0, :]
        passive_counts_flat = passive_counts.reshape(-1, passive_counts.shape[-2] * passive_counts.shape[-1])
        passive_counts_flat /= passive_counts_flat.sum(-1, keepdims=True) # Normalize
        #   Run k-means clustering
        cluster_assignments_flat = KMeans(n_clusters=clusters).fit_predict(passive_counts_flat)
        cluster_assignments = cluster_assignments_flat.reshape(passive_counts.shape[:-2])
        #   Get the cluster level counts
        cluster_counts = np.stack([count_matrices[cluster_assignments == cluster].sum(0) for cluster in range(clusters)])
        #  Add the population level prior to each cluster
        cluster_counts += prior_coefficient * pop_level_count[None, ...]
        cluster_counts_copy = np.copy(cluster_counts)
        cluster_counts /= cluster_counts.sum(-1, keepdims=True) # Normalize to get probabilities
        prior = cluster_counts[cluster_assignments_flat].reshape(count_matrices.shape)
        prior_coefficient = 1
        count_matrices += prior_coefficient * prior
        count_matrices /= count_matrices.sum(-1, keepdims=True)
        # Making sure P(state=1) > P(state=0) for active and passive actions
        invalid_state_action_0 = set()
        invalid_state_action_1 = set()
        for i in range(clusters):
            if cluster_counts[i, 1, 0, 1] <= cluster_counts[i, 0, 0, 1]:
                invalid_state_action_0.add(i)
            if cluster_counts[i, 1, 1, 1] <= cluster_counts[i, 0, 1, 1]:
                invalid_state_action_1.add(i)
        delta_0 = 0
        delta_1 = 0
        num_valid_clusters_0 = 0
        num_valid_clusters_1 = 0
        
        for i in range(clusters):
            if i not in invalid_state_action_0:
                delta_0 += cluster_counts[i, 1, 0, 1] - cluster_counts[i, 0, 0, 1]
                num_valid_clusters_0+=1
            if i not in invalid_state_action_1:
                delta_1 += cluster_counts[i,1,1,1] - cluster_counts[i,0, 1, 1]
                num_valid_clusters_1+=1
        delta_0 = delta_0/num_valid_clusters_0
        delta_1 = delta_1/num_valid_clusters_1

        for i in range(count_matrices.shape[1]):
            if count_matrices[0,i,1, 0, 1] <= count_matrices[0,i,0,0,1]:
                count_matrices[0,i,1, 0, 1] = min(count_matrices[0,i,0,0,1] + delta_0,1)
            if count_matrices[0,i,1,1,1] <= count_matrices[0,i, 0, 1, 1]:
                count_matrices[0,i,1,1,1] = min(count_matrices[0,i, 0, 1, 1] + delta_1,1)
        count_matrices[0,:,:, :, 0] = 1 - count_matrices[0,:,:, :, 1]
        
        
        # Making sure P(action=1) > P(action=0) for all states
        
        # keep record of all the clusters that currently have invalid active probabilities
        # a cluster is invalid if it (1) has a lower active probability than passive, or (2) it has less than 7 counts in that state
        # we need to keep track separately for 0 as a starting state versus 1
        invalid_active_set_state_0 = set()
        invalid_active_set_state_1 = set()
        for i in range(clusters):
            if np.sum(cluster_counts_copy[i, 0, 1, :]) < 7 or cluster_counts[i, 0, 1, 1] <= cluster_counts[i, 0, 0, 1]:
                invalid_active_set_state_0.add(i)
            if np.sum(cluster_counts_copy[i, 1, 1, :]) < 7 or cluster_counts[i, 1, 1, 1] <= cluster_counts[i, 1, 0, 1]:
                invalid_active_set_state_1.add(i)
        delta_0 = 0
        delta_1 = 0
        num_valid_clusters_0 = 0
        num_valid_clusters_1 = 0
        # Calculate delta = average difference between active and passive probabilities for all clusters that are valid
        # there is a separate delta for 0 as a starting state and 1.
        for i in range(clusters):
            if i not in invalid_active_set_state_0:
                delta_0 += cluster_counts[i,0,1,1] - cluster_counts[i,0,0,1]
                num_valid_clusters_0+=1
            if i not in invalid_active_set_state_1:
                delta_1 += cluster_counts[i,1,1,1] - cluster_counts[i,1,0,1]
                num_valid_clusters_1+=1
        delta_0 = delta_0/num_valid_clusters_0
        delta_1 = delta_1/num_valid_clusters_1
        # for all arms with active<passive probability, add active = passive + delta
        for i in range(count_matrices.shape[1]):
            if count_matrices[0,i,0,1,1] - count_matrices[0,i,0,0,1]<=0.01:
                count_matrices[0,i,0,1,1] = min(count_matrices[0,i,0,0,1] + delta_0,1)
            if count_matrices[0,i,1,1,1] <= count_matrices[0,i,1,0,1]:
                count_matrices[0,i,1,1,1] = min(count_matrices[0,i,1,0,1] + delta_1,1)        
        count_matrices[0,:,:,:, 0] = 1 - count_matrices[0,:,:,:, 1]
        
        # a few small bandaids
        for i in range(count_matrices.shape[1]):
            if count_matrices[0,i,1,1,1]<=count_matrices[0,i,0,1,1] +0.03:
                count_matrices[0,i,1,1,1] = min(count_matrices[0,i,0,1,1] + 0.1,1)
        for i in range(count_matrices.shape[1]):
            if count_matrices[0,i,1,1,1]<=count_matrices[0,i,1,0,1] +0.03:
                count_matrices[0,i,1,1,1] = min(count_matrices[0,i,1,0,1] + 0.1,1)
        for i in range(count_matrices.shape[1]):
            if count_matrices[0,i,1,0,1]<=count_matrices[0,i,0,0,1]+0.03:
                count_matrices[0,i,1,0,1] = min(count_matrices[0,i,0,0,1] + 0.1,1)
        for i in range(count_matrices.shape[1]):
            if count_matrices[0,i,0,1,1]<=count_matrices[0,i,0,0,1]+0.03:
                count_matrices[0,i,0,1,1] = min(count_matrices[0,i,0,0,1] + 0.1,1)
    return count_matrices,None

def get_count_matrices(
    trajs: np.ndarray
):
    """
    Given a set of RMAB trajectories, count the number of s,a,s' transitions for each beneficiary.
    """
    # Shape constants 
    num_states = trajs[..., Dataset.tuple_dict['state']].max().astype(int) + 1  # Add 1 because states are 0-indexed # type: ignore
    num_actions = trajs[..., Dataset.tuple_dict['action']].max().astype(int) + 1 # type: ignore

    # Create count matrices of the appropriate shape
    #   Note: scatter_nd is used to count the number of times each (s,a,s') tuple occurs in each trajectory. It is magic.
    # Flatten trajectories to make it easier to count
    batch_shape = trajs.shape[:-2]
    trajs_flat = trajs.reshape(-1, trajs.shape[-2], trajs.shape[-1])
    num_trajs = trajs_flat.shape[0]

    # Create count matrices of the appropriate shape
    count_matrices = np.zeros((trajs_flat.shape[0], num_states, num_actions, num_states), dtype=np.int64)
    num_timesteps = trajs_flat.shape[-2]
    for t in range(num_timesteps):
        count_matrices[np.arange(num_trajs), trajs_flat[:, t, Dataset.tuple_dict['state']], trajs_flat[:, t, Dataset.tuple_dict['action']], trajs_flat[:, t, Dataset.tuple_dict['next_state']]] += 1

    # Reshape count matrices to have the same batch shape as the input
    count_matrices = count_matrices.reshape(batch_shape + count_matrices.shape[-3:])

    return count_matrices

# Extract the budget from the trajectories
def extract_budget(offline_traj: np.ndarray) -> int:
    # Get the budget for each trajectory
    offline_traj = offline_traj
    actions = offline_traj[..., Dataset.tuple_dict["action"]]
    budget_per_traj = actions.mean(axis=-1).sum(axis=-1)  # Take mean across timesteps and sum across beneficiaries
    return budget_per_traj.mean().round().astype(int)
    
def split_offline_traj(offline_traj: np.ndarray, num_trajectories: int):
    # Get beneficiaries who were intervened at least once
    ever_intervened = (offline_traj[:, :, 1].sum(axis=-1) > 0).astype(bool)
    num_benefs = ever_intervened.shape[0]
    benef_idx_interv = np.arange(num_benefs)[ever_intervened]
    benef_idx_not_interv = np.arange(num_benefs)[~ever_intervened]

    # Make sure that these arrays are a multiple of num_trajectories
    def _make_multiple_of_num_trajectories(arr: np.ndarray, num_trajectories: int):
        num_benefs = arr.shape[0]
        num_benefs_per_traj = num_benefs // num_trajectories
        num_benefs = num_benefs_per_traj * num_trajectories
        arr = arr[:num_benefs]
        return arr
    benef_idx_interv = _make_multiple_of_num_trajectories(benef_idx_interv, num_trajectories)
    benef_idx_not_interv = _make_multiple_of_num_trajectories(benef_idx_not_interv, num_trajectories)

    # Shuffle the indices
    np.random.shuffle(benef_idx_interv)
    np.random.shuffle(benef_idx_not_interv)

    # Split the beneficiaries into num_trajectories trajectories
    benef_idx_interv_split = np.split(benef_idx_interv, num_trajectories)
    benef_idx_not_interv_split = np.split(benef_idx_not_interv, num_trajectories)
    combined_idx_split = [np.concatenate([interv, not_interv]) for interv, not_interv in zip(benef_idx_interv_split, benef_idx_not_interv_split)]  # type: ignore

    return combined_idx_split

# If run as a script, run the unit tests
if __name__=='__main__':
    pass
    # Test get_count_matrices
    # traj = np.random.uniform(0, 2, (2, 10, 5, 4)).astype('int64')

    # count_matrices = get_count_matrices(traj_rmab)
    # assert count_matrices.shape == (2, 10, 2, 2, 2)
    # assert count_matrices.numpy().sum() == 2 * 10 * 5  # type: ignore

    # Test extract_mdps
    # prob_matrices = extract_mdps(traj)
    # assert prob_matrices.shape == (2, 10, 2, 2, 2)
    # assert prob_matrices.numpy().sum() == 2 * 4 * 5
    # assert prob_matrices.numpy().min() >= 0
    # assert prob_matrices.numpy().max() <= 1
    # assert (prob_matrices.numpy().sum(axis=-1) == 1).all()
