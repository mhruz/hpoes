import os
import io
import argparse
import sys
import numpy as np
import h5py
import cupy
from chainer import serializers
from chainer.cuda import to_gpu, to_cpu
import chainer
from .architecture_chainer_CNN3D_V2V_PoseNet import *
from . import v2v_misc
from . import binvox_rw
import time

if __name__ == '__main__':
    # parse commandline
    parser = argparse.ArgumentParser(description='Test the V2V-PoseNet.')
    parser.add_argument('test_h5', type=str, help='path to testing h5 file')
    parser.add_argument('network', type=str, help='network to test (h5 file)')
    parser.add_argument('output_file', type=str, help='output file (h5 file)')
    parser.add_argument('--batch_size', type=int, help='batch size in each iteration, default value = 8', default=8)
    parser.add_argument('--joint_idxs', type=int, nargs='+', help='list of indexes of joints to test on, defaul = all joints are evaluated')
    parser.add_argument('--epoch_ensemble', type=bool, help='whether to use epoch ensemble (the other nets should end with "_epoch_X.h5"')
    parser.add_argument('--epoch_ensemble_type', type=str, help='type of epoch ensemble to use, either \"avg\" or \"max\"', default='avg')
    parser.add_argument('--num_joints', type=int, help='you can specify how many joints is the net predicting')
    parser.add_argument('--data_label', type=str, help='the data label in the H5 file, default = real_voxels', default='real_voxels')
    parser.add_argument('--data_range', type=str, help='the processed data range in the form of the string: \'start_index:end_index\', default value = 0:, which means process all data', default='0:')
    parser.add_argument('--output_error', type=bool, help='whether to use error computation')
    parser.add_argument('--number_of_gpus', type=int, help='name of GPUs to use for computing, default = 1', default=1)

    args=parser.parse_args()
    args_dict = args.__dict__
    for arg in sorted(args_dict.keys()):
        print(arg, args_dict[arg])

    batch_size = args.batch_size

    # check if the batch size is divisible by number of gpus
    if batch_size % args.number_of_gpus != 0:
        print('Batch size is not divisible by the number of GPUs. Exiting.')
        sys.exit(0)

    # read all the networks for ensemblement
    if args.epoch_ensemble is not None:
        path_to_nets = os.path.dirname(args.network)
        nets_fn = os.listdir(path_to_nets)

        nets_fn = [os.path.join(path_to_nets, x) for x in nets_fn if x.endswith('.h5') and '_epoch_' in x]
    else:
        nets_fn = [args.network]

    # set the data key in H5
    key = args.data_label

    # open the H5 file for testing
    f = h5py.File(args.test_h5, 'r')

    # determine the number of joints
    if args.num_joints is None:
        numJoints = f['labels'].shape[1]
    else:
        numJoints = args.num_joints

    errors = []
    errors_max = []
    errors_smooth_max = []
    npoints = []

    # create multiple models on multiple GPUs
    models = [V2V(numJoints)]

    # distribute the models on the GPUs
    for i in range(1, args.number_of_gpus):
        models.append(models[0].copy())

    for i, m in enumerate(models):
        try:
            m.to_gpu(i)
        except cupy.cuda.runtime.CUDARuntimeError:
            print('Invalid number of GPUs specified. Error on GPU id {}.'.format(i))
            sys.exit(0)

    loaded_net = ''
    # the output H5 file
    h5_file = h5py.File(args.output_file, 'w')
    
    # for time measure
    s = time.time()
    
    with chainer.using_config('train', False):
        data_range = [0, len(f[key])]
        if args.data_range != '0:':
            for i,x in enumerate(args.data_range.split(':')):
                if x != '':
                    data_range[i] = int(x)
        j = 0
        for i in range(data_range[0], data_range[1], batch_size):
            print(i, '/', data_range[1])

            batch_data = []
            batch_labels = []
            cubes = []
            upper_range = min(batch_size, data_range[1] - i)
            for idx in range(0, upper_range):
                batch_data_frame = []

                sdata = f[key][str(idx + i)][:].tostring()
                _file = io.BytesIO(sdata)
                data = binvox_rw.read_as_3d_array(_file).data
                data = data.astype(np.float32)
                batch_data_frame.append(data)
                    
                batch_data.append(batch_data_frame)

                if args.output_error:
                    if args.joint_idxs is None or f['labels'].shape[1] == args.num_joints:
                        batch_labels.append(f['labels'][idx + i])
                    else:
                        batch_labels.append(f['labels'][idx + i, args.joint_idxs, :])

                cubes.append(f['cubes'][idx + i])

            batch_data = np.array(batch_data, dtype=np.float32)
            batch_labels = np.array(batch_labels, dtype=np.float32)

            # divide the data for individual GPUs
            batch_data_gpu = []

            one_gpu_load = batch_data.shape[0] // args.number_of_gpus
            for g in range(args.number_of_gpus):
                batch_data_gpu.append(batch_data[g * one_gpu_load:(g + 1) * one_gpu_load])
                
            if one_gpu_load * args.number_of_gpus < batch_data.shape[0]:
                diff = batch_data.shape[0] - one_gpu_load * args.number_of_gpus
                batch_data_gpu[-1] = np.append(batch_data_gpu[-1], batch_data[-diff:], axis=0)

            # if no network is loaded, determine the output shape
            if loaded_net == '':
                for m in models:
                    serializers.load_hdf5(nets_fn[0], m)

                loaded_net = nets_fn[0]

                M = models[0]

                pred = M(to_gpu(np.asarray([batch_data[0]])))
                output_shape = pred.data.shape

            # the data will be aggregated from different GPUs into this variable
            data_aggregate = np.zeros((batch_data.shape[0], numJoints, output_shape[2], output_shape[3], output_shape[4]))

            for fn in nets_fn:
                if loaded_net != fn:
                    for n, m in enumerate(models):
                        serializers.load_hdf5(fn, m)

                predictions = []
                for n, m in enumerate(models):
                    device_data = chainer.Variable(to_gpu(batch_data_gpu[n], device=n))
                    
                    pred = m(device_data).data
                    predictions.append(pred)

                for g, pred in enumerate(predictions):
                    if args.epoch_ensemble_type == 'avg':                        
                        if one_gpu_load < pred.shape[0]:
                            data_aggregate[g*one_gpu_load:] += to_cpu(pred)
                        else:
                            data_aggregate[g*one_gpu_load:(g + 1)*one_gpu_load] += to_cpu(pred)
                    elif args.epoch_ensemble_type == 'max':
                        if one_gpu_load < pred.shape[0]:
                            data_aggregate[g * one_gpu_load:(g + 1) * one_gpu_load] = np.maximum(data_aggregate[g * one_gpu_load:(g + 1) * one_gpu_load], to_cpu(pred))
                        else:
                            data_aggregate[g * one_gpu_load:] = np.maximum(data_aggregate[g * one_gpu_load:], to_cpu(pred))

                loaded_net = fn

            if args.epoch_ensemble_type == 'avg':
                data_aggregate /= len(nets_fn)
            
            if (args.joint_idxs is None) or (args.num_joints == data_aggregate.shape[1]):
                points3D_pred = v2v_misc.heatMaps2Points3D(data_aggregate, cubes)
                points3D_pred_max = v2v_misc.heatMaps2Points3D_max(data_aggregate, cubes)
                points3D_pred_smooth_max, conf_pred_smooth_max = v2v_misc.heatMaps2Points3D_smooth_max(data_aggregate, cubes, smooth_size=3, Nvox=output_shape[2])
            else:
                points3D_pred = v2v_misc.heatMaps2Points3D(data_aggregate[:, args.joint_idxs, :, :, :], cubes)
                points3D_pred_max = v2v_misc.heatMaps2Points3D_max(data_aggregate[:, args.joint_idxs, :, :, :], cubes)
                points3D_pred_smooth_max, conf_pred_smooth_max = v2v_misc.heatMaps2Points3D_smooth_max(data_aggregate[:, args.joint_idxs, :, :, :], cubes, smooth_size=3, Nvox=output_shape[2])

            if args.output_error:
                errors.append(v2v_misc.meanErrorOnHand(batch_labels, points3D_pred))
                errors_max.append(v2v_misc.meanErrorOnHand(batch_labels, points3D_pred_max))
                errors_smooth_max.append(v2v_misc.meanErrorOnHand(batch_labels, points3D_pred_smooth_max))
                npoints.append(batch_data.shape[0])
            
            if i == data_range[0]:
                print('Detected Nvox_labels:', output_shape[2])
                num_items = data_range[1] - data_range[0]
                points3D_pred_shape = points3D_pred.shape
                h5_file.create_dataset(name="points3D_pred", shape=(num_items, points3D_pred_shape[1], points3D_pred_shape[2]), dtype=np.float32)
                points3D_pred_max_shape = points3D_pred_max.shape
                h5_file.create_dataset(name="points3D_pred_max", shape=(num_items, points3D_pred_max_shape[1], points3D_pred_max_shape[2]), dtype=np.float32)
                points3D_pred_smooth_max_shape = points3D_pred_smooth_max.shape
                h5_file.create_dataset(name="points3D_pred_smooth_max", shape=(num_items, points3D_pred_smooth_max_shape[1], points3D_pred_smooth_max_shape[2]), dtype=np.float32)
                conf_pred_smooth_max_shape = conf_pred_smooth_max.shape
                h5_file.create_dataset(name="conf_pred_smooth_max", shape=(num_items, conf_pred_smooth_max_shape[1]), dtype=np.float32)

            start = j
            end = j + batch_data.shape[0]
            h5_file["points3D_pred"][start:end] = points3D_pred
            h5_file["points3D_pred_max"][start:end] = points3D_pred_max
            h5_file["points3D_pred_smooth_max"][start:end] = points3D_pred_smooth_max
            h5_file["conf_pred_smooth_max"][start:end] = conf_pred_smooth_max
            j += batch_data.shape[0]
    
    f.close()
    h5_file.close()
    
    print(time.time() - s)

    if args.output_error:
        # compute the errors from the aggregated data
        error = 0
        error_max = 0
        error_smooth_max = 0
        total_points = 0
        for (err, err_max, err_smooth, w) in zip(errors, errors_max, errors_smooth_max, npoints):
            error += w*err
            error_max += w * err_max
            error_smooth_max += w * err_smooth
            total_points += w

        error /= total_points
        error_max /= total_points
        error_smooth_max /= total_points

        print('weighted avg error: ', error)
        print('maximum error: ', error_max)
        print('smooth maximum error: ', error_smooth_max)