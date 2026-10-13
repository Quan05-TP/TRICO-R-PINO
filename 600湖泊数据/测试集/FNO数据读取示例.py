"""Read prepared temperature fields without treating missing cells as observed zeros."""
from pathlib import Path
import numpy as np

def load_profiles(npz_path):
    with np.load(npz_path, allow_pickle=False) as source:
        data={key:source[key] for key in source.files}
    temperature=data['temperature_grid_c']
    mask=data['supported_mask']
    depth=np.broadcast_to(data['depth_normalized'],temperature.shape)
    # Layout compatible with a 1D FNO: samples, channels, depth.
    # Channels: temperature, support indicator, normalized depth.
    data['profile_features']=np.stack([np.where(mask,temperature,0),mask.astype(float),depth],axis=1).astype('float32')
    return data

def mse_at_original_depths(prediction_c, data):
    """Evaluate predicted grid values against original measurements, not interpolated targets.

    prediction_c has shape (number_of_profiles, number_of_depth_nodes).
    This NumPy evaluation helper is not an autograd training loss.
    """
    prediction=np.asarray(prediction_c)
    if prediction.shape!=data['temperature_grid_c'].shape:
        raise ValueError('prediction shape must match temperature_grid_c')
    squared=[]
    for i in range(len(prediction)):
        start,end=data['observation_offsets'][i:i+2]
        z=data['observed_depth_m'][start:end]
        observed=data['observed_temperature_c'][start:end]
        predicted=np.interp(z,data['depth_grid_m'],prediction[i])
        valid=np.isfinite(predicted)&np.isfinite(observed)
        squared.extend(((predicted[valid]-observed[valid])**2).tolist())
    return float(np.mean(squared)) if squared else float('nan')

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('npz_file')
    args=parser.parse_args();data=load_profiles(Path(args.npz_file))
    print('lake_id:',str(data['lake_id']))
    print('profile_features:',data['profile_features'].shape)
    print('thermocline_profiles:',int(data['thermocline_detected'].sum()))
