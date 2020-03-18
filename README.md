# hpoes
The Hand Pose Estimation framework

###### **Data**

The data are expected in a H5 file format. The file is expected to have the following structure:

>>> import h5py
>>> f = h5py.File('filename.h5', 'r')
>>> f['real_voxels']
<HDF5 group "/real_voxels" (N members)>
Where N is the number of test samples. The key 'real_voxels' can be changed via the script's 'test_net.py' argument.
>>> f['real_voxels']['0']
<HDF5 dataset "0": shape (1,), type "|V8566">
Each test sample is of different length since it is represented by 'binvox_rw.py' as a compressed string. 
>>> f['cubes']
<HDF5 dataset "cubes": shape (N, 3), type "<f4">
Where N is the number of test samples and for each sample there are 3D sizes of the cube containing the hand in millimeters.
Only cubes are supported for now (all three dimensions are the same).