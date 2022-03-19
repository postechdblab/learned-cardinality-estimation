import numpy as np

def get_batch_job(batch_id, directory):
    target_cost_batch = np.load(directory+'/target_cost_'+str(batch_id)+'.np.npy')
    target_cardinality_batch = np.load(directory+'/target_cardinality_'+str(batch_id)+'.np.npy')
    operators_batch = np.load(directory+'/operators_'+str(batch_id)+'.np.npy')
    extra_infos_batch = np.load(directory+'/extra_infos_'+str(batch_id)+'.np.npy')
    condition1s_batch = np.load(directory+'/condition1s_'+str(batch_id)+'.np.npy')
    condition2s_batch = np.load(directory+'/condition2s_'+str(batch_id)+'.np.npy')
    samples_batch = np.load(directory+'/samples_'+str(batch_id)+'.np.npy')
    condition_masks_batch = np.load(directory+'/condition_masks_'+str(batch_id)+'.np.npy')
    mapping_batch = np.load(directory+'/mapping_'+str(batch_id)+'.np.npy')
    return target_cost_batch, target_cardinality_batch, operators_batch, extra_infos_batch, condition1s_batch,\
           condition2s_batch, samples_batch, condition_masks_batch, mapping_batch
