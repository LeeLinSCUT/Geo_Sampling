import tensorflow as tf
from tensorflow.python.framework import ops
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
test_module=tf.load_op_library(os.path.join(BASE_DIR, 'sample.so'))


def query_ball_point(radius, nsample, xyz1):
    '''
    Input:
        radius: float32, ball search radius
        nsample: int32, number of points selected in each ball region
        xyz1: (batch_size, ndataset, 3) float32 array, input points
    Output:
        idx: (batch_size, npoint, nsample) int32 array, indices to input points
    '''
    #return grouping_module.query_ball_point(radius, nsample, xyz1, xyz2)
    return test_module.query_ball_point(xyz1, radius, nsample)
ops.NoGradient('QueryBallPoint')

def sampledemo(inp):
#input:(b,2048,8,3)
#output(b,2048)
    return test_module.sampledemo(inp) # op 的名字是ZeroOut，那么在python 中，名字是 zero_out
ops.NoGradient('Sampledemo') #首字母大写

def knndemo(inp):
#input:(b,2048,8,3)
#output(b,2048)
    return test_module.knndemo(inp) # op 的名字是ZeroOut，那么在python 中，名字是 zero_out
ops.NoGradient('Knndemo') #首字母大写

def cubeselect(xyz, radius):
    """
    :param xyz: (b, n, 3) float
    :param radius: float
    :return: (b, n, 8) int
    """
    idx = test_module.cube_select(xyz, radius)
    return idx
ops.NoGradient('CubeSelect')
def farthest_point_sample(npoint,inp):
    '''
input:
    int32
    batch_size * ndataset * 3   float32
returns:
    batch_size * npoint         int32
    '''
    return test_module.farthest_point_sample(inp, npoint)
ops.NoGradient('FarthestPointSample')

def gather_point(xyz, idx):
    """
    xyz:(b,n,3)
    idx:(b,nsamples)
    out:(b,nsamples,3)
    """
    out = test_module.gather_point(xyz, idx)
    return out
ops.NoGradient('GatherPoint')


def gather_pointfar(inp,idx):
    '''
input:
    batch_size * ndataset * 3   float32
    batch_size * npoints        int32
returns:
    batch_size * npoints * 3    float32
    '''
    return test_module.gather_pointfar(inp,idx)
#@tf.RegisterShape('GatherPoint')
#def _gather_point_shape(op):
#    shape1=op.inputs[0].get_shape().with_rank(3)
#    shape2=op.inputs[1].get_shape().with_rank(2)
#    return [tf.TensorShape([shape1.dims[0],shape2.dims[1],shape1.dims[2]])]
@tf.RegisterGradient('GatherPointfar')
def _gather_point_grad(op,out_g):
    inp=op.inputs[0]
    idx=op.inputs[1]
    return [test_module.gather_point_grad(inp,idx,out_g),None]

def group_point(points, idx):
    '''
    Input:
        points: (batch_size, ndataset, channel) float32 array, points to sample from
        idx: (batch_size, npoint, nsample) int32 array, indices to points
    Output:
        out: (batch_size, npoint, nsample, channel) float32 array, values sampled from points
    '''
    return test_module.group_point(points, idx)
@tf.RegisterGradient('GroupPoint')
def _group_point_grad(op, grad_out):
    points = op.inputs[0]
    idx = op.inputs[1]
    return [test_module.group_point_grad(points, idx, grad_out), None]
#    grouped_xyz = grouped_xyz - tf.tile(xyz[:,:,None,:],[1,1,8,1]) #(b_s,8192,8,3) - (b_s,8192,8,3)

if __name__=='__main__':
	import numpy as np
	import h5py
	f = h5py.File("Area_4_hallway_10.h5",'r')
	data = f["data"][:][2]  #(2048,3)
	choose = np.random.choice(1536,512,replace=False)
	with tf.desvice('/gpu:0'):
		inp = tf.constant(data)     
		inp1 = tf.expand_dims(inp,0) #(1,2048,3)
		new_xyz = gather_pointfar(inp1, farthest_point_sample(1024, inp1))
		ones = np.ones([2048])
		idx = knndemo(inp1) #(1,2048,8)
		xyz = group_point(inp1,idx) #(1,2048,8,3)
		xyz1 = xyz - tf.tile(inp1[:,:,None,:],[1,1,6,1]) #(1,2048,8,3)
		dot_ori = sampledemo(xyz1) #（1，2048)
		sort = tf.argsort(dot_ori,axis=-1,direction='ASCENDING') #(1,2048)
		sort = tf.to_int64(sort)
	with tf.device('/cpu:0'):
		sort1,sort2 = tf.split(sort,[1536,512],axis=-1) #(1,1536) (1,512)
	with tf.device('/gpu:0'):
		choose_tensor= tf.convert_to_tensor(choose)
		sort1_out = tf.gather(sort1,choose,axis=-1) #(1,512)
		out_final = tf.concat((sort2,sort1_out),axis=-1)#(1.1024)
		out_32 = tf.to_int32(out_final)
		out = gather_point(inp1,out_32)
