import numpy as np
import csv
import glob
import scipy.io as sio

folder = "pdr_data/hallway"
stats = []
saved_trajectories = {}
best_detector_count = {'shoe': 0, 'ared': 0, 'amvd': 0, 'mbgtd':0, 'vicon':0}

for f in glob.glob('{}/*.mat'.format(folder)):
    print("Processing {}".format(f))
    trial_name = f.replace(folder,'')
    trial_stats = [trial_name]
    print(trial_name)
    
    data = sio.loadmat(f)
    for key in data:
        print(key)
        print(data[key])
    imu = data['imu']
    ts = data['ts']
    print(ts[0])
    gt = data['gt']
    #best_detector = data['best_detector'][0]
    print(imu[0])
    print(gt[0])
    #print(best_detector)
    #zv_shoe_opt = data['zv_shoe_opt'][0]
    #print(zv_shoe_opt)
    csv_filename = f.replace("processed_data.mat","imu_data_clean.csv")
    with open(csv_filename, "w") as f:
        writer = csv.writer(f)
        writer.writerow("ax,ay,az,wx,wy,wz,px,py,pz")
        #for im,g,d in zip(imu,gt,zv_shoe_opt):
        #    writer.writerow(np.r_[im,g,d])
        #writer.writerow("ax,ay,az,wx,wy,wz,px,py,pz")
        for im in imu:
            writer.writerow(im)
    
