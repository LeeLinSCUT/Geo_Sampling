#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include<stdio.h>


// input: radius (1), nsample (1), xyz1 (b,n,3)
// output: idx (b,n,nsample)  
__global__ void query_ball_point_gpu(int b, int n, float radius, int nsample, const float *xyz1, int *idx) {
	int batch_idx = blockIdx.x;
	xyz1 +=batch_idx*n*3;
	idx += batch_idx*n*nsample;
	float judge_radius = radius * radius;
    for (int j=threadIdx.x;j<n;j+=blockDim.x) {
        int cnt = 0;
       	for (int l=0;l<nsample;++l)
         	idx[j*nsample+l] = j;
        float x1=xyz1[j*3+0];
        float y1=xyz1[j*3+1];
        float z1=xyz1[j*3+2]; 
        for (int k=0;k<n;++k) {   
            if (cnt == nsample)  //如果采集够点了
                break; // only pick the FIRST nsample points in the ball
            if (k==j)
            {
            	continue;
            }
            float x2=xyz1[k*3+0];
            float y2=xyz1[k*3+1];
            float z2=xyz1[k*3+2];
    	    float d=(x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1);
            if (d<judge_radius) {
                idx[j*nsample+cnt] = k;
                cnt+=1;
            }
        }
    }
}
/*
__global__ void sampleKernel(int b, int n,const float *inp, float *result)  //<<<1,512>>> //inp:b*2048*8*3  out:(b,2048)
{
	float dot=0;
	int batch_idx = blockIdx.x;     //处理的batch_idx 
	inp +=batch_idx*n*24;
	for(int i=threadIdx.x;i<n;i+=blockDim.x) //每个block处理一个batch 每个thread处理一组点 i:0 512 1024 1536 
	{
        float temp_dot[8];
		result[batch_idx*n+i] = 0;
		for(int j=0;j<8;j++) //处理一个点
		{
            float x =inp[i*24+j*3+0];
            float y =inp[i*24+j*3+1];
            float z =inp[i*24+j*3+2];
            for(int num=j;num>-1;num--)
            {
            float x1=inp[i*24+num*3+0];
            float y1=inp[i*24+num*3+1];
            float z1=inp[i*24+num*3+2];
            dot = (x*x1+y*y1+z*z1);
            temp_dot[j] += dot;
            temp_dot[num] += dot;
            }
		}
        result[batch_idx*n+i] = temp_dot[0];
        for(int num=1;num<8;num++)
        {
            if(result[batch_idx*n+i]<temp_dot[num])
                result[batch_idx*n+i] = temp_dot[num];
        }

	}
}*/

__global__ void sampleKernel(int b, int n,const float *inp, float *result)  //<<<1,512>>> //inp:b*2048*8*3  out:(b,2048)
{
    float dot=0;
    int batch_idx = blockIdx.x;     //处理的batch_idx 
    float temp_dist = 0;
    float temp_dist1 = 0;
    inp +=batch_idx*n*18;
    for(int i=threadIdx.x;i<n;i+=blockDim.x) //每个block处理一个batch 每个thread处理一组点 i:0 512 1024 1536 
    {
        result[batch_idx*n+i] = 0;
        float x=0;
        float y=0;
        float z=0;
        for(int j=0; j<6;j++)
        {
            x +=inp[i*18+j*3+0];
            y +=inp[i*18+j*3+1];
            z +=inp[i*18+j*3+2];
        }
        x =x/6;
        y =y/6;
        z =z/6;
        for(int j=0;j<6;j++) //处理一个点
        {
            float x1=inp[i*18+j*3+0];
            float y1=inp[i*18+j*3+1];
            float z1=inp[i*18+j*3+2];
            temp_dist1 = (x1*x1+y1*y1+z1*z1);
            if(temp_dist1 != 0)
            {
                dot = (x*x1+y*y1+z*z1)/temp_dist1;
                result[batch_idx*n+i] += dot;
            }
        }   
    }

}    

__global__ void knnKernel(int b, int n,const float *xyz, int *idx_out)  //<<<1,512>>> //inp:b*2048*3  out:(b,2048,6)
{
    int batch_idx = blockIdx.x;     //处理的batch_idx 
    xyz +=batch_idx*n*3;
    idx_out +=batch_idx*n*6;
    for(int i=threadIdx.x;i<n;i+=blockDim.x) //每个block处理一个batch 每个thread处理一组点 i:0 512 1024 1536 
    {
        float temp_dist[6]={1e8,1e8,1e8,1e8,1e8,1e8};
        int first_idx = 0;
        float x=xyz[i*3+0];
        float y=xyz[i*3+1];
        float z=xyz[i*3+2];
        for(int j=0;j<n;j++)
        {
            if(i==j) continue;
            float tx = xyz[j * 3];
            float ty = xyz[j * 3 + 1];
            float tz = xyz[j * 3 + 2];
            float dist = (x - tx) * (x - tx) + (y - ty) * (y - ty) + (z - tz) * (z - tz);
            if(dist < temp_dist[first_idx]) 
            {
                idx_out[i*6+first_idx] = j;
                temp_dist[first_idx] = dist;    
                for(int num=0;num<6;num++)
                {
                    if(temp_dist[first_idx]<temp_dist[num])
                        first_idx = num;
                }
            }
        }
    }
}


__global__ void cube_select(int b, int n,float radius, const float* xyz, int* idx_out) {
    int batch_idx = blockIdx.x;     //b:batch_size n:points_num radius: r xyz:(b,n,3) idx_out(b,n,8)
    xyz += batch_idx * n * 3;       //指向指定要处理的批次 每个block处理一个batch 
    idx_out += batch_idx * n * 8;
    float temp_dist[8];     
    float judge_dist = radius * radius;
    for(int i = threadIdx.x; i < n;i += blockDim.x) { //这里处理一个批次，如果有1024个点 则每个线程会重复这个循环两次 
        float x = xyz[i * 3];      //指定点
        float y = xyz[i * 3 + 1];
        float z = xyz[i * 3 + 2];
        for(int j = 0;j < 8;j ++) {
            temp_dist[j] = 1e8; //初始化距离
            idx_out[i * 8 + j] = i; // if not found, just return itself..  //这里先默认把索引点为自己
        }
        for(int j = 0;j < n;j ++) {    //遍历该batch下的所有点
            if(i == j) continue;
            float tx = xyz[j * 3];
            float ty = xyz[j * 3 + 1];
            float tz = xyz[j * 3 + 2];
            float dist = (x - tx) * (x - tx) + (y - ty) * (y - ty) + (z - tz) * (z - tz);
            if(dist > judge_dist) continue; //如果距离超过规定的距离 继续循环
            int _x = (tx > x);              //
            int _y = (ty > y);
            int _z = (tz > z);
            int temp_idx = _x * 4 + _y * 2 + _z; //存放的位置
            if(dist < temp_dist[temp_idx]) {
                idx_out[i * 8 + temp_idx] = j;
                temp_dist[temp_idx] = dist;
            }
        }
    }
}

__global__ void group_point_gpu(int b, int n, int c, int m, int nsample, const float *points, const int *idx, float *out) {
    int batch_index = blockIdx.x;
    points += n*c*batch_index;
    idx += m*nsample*batch_index;
    out += m*nsample*c*batch_index;

    int index = threadIdx.x;
    int stride = blockDim.x;
    
    for (int j=index;j<m;j+=stride) {
        for (int k=0;k<nsample;++k) {
            int ii = idx[j*nsample+k];
            for (int l=0;l<c;++l) {
                out[j*nsample*c+k*c+l] = points[ii*c+l];
            }
        }
    }
}
__global__ void gather_point(int b, int n, int nsamples, const int* idx, const float* xyz, float* result) {   //idx(b,nsamples) xyz(b,n,3) result(b,nasmples,3)
    int batch_index = blockIdx.x;
    idx += batch_index*nsamples;
    xyz += batch_index*n*3;
    result += batch_index*nsamples*3;
    for(int i = threadIdx.x; i<nsamples;i +=blockDim.x)
    {   
        int j = idx[i];
        result[i*3+0] = xyz[j*3];
        result[i*3+1] = xyz[j*3+1];
        result[i*3+2] = xyz[j*3+2];
    }

}

__global__ void gatherpointKernel(int b,int n,int m,const float * __restrict__ inp,const int * __restrict__ idx,float * __restrict__ out){
  for (int i=blockIdx.x;i<b;i+=gridDim.x){
    for (int j=blockIdx.y*blockDim.x+threadIdx.x;j<m;j+=blockDim.x*gridDim.y){
      int a=idx[i*m+j];
      out[(i*m+j)*3+0]=inp[(i*n+a)*3+0];
      out[(i*m+j)*3+1]=inp[(i*n+a)*3+1];
      out[(i*m+j)*3+2]=inp[(i*n+a)*3+2];
    }
  }
}

__global__ void farthestpointsamplingKernel(int b,int n,int m,const float * __restrict__ dataset,float * __restrict__ temp,int * __restrict__ idxs){
  if (m<=0)
    return;
  const int BlockSize=512;
  __shared__ float dists[BlockSize];
  __shared__ int dists_i[BlockSize];
  const int BufferSize=3072;
  __shared__ float buf[BufferSize*3];
  for (int i=blockIdx.x;i<b;i+=gridDim.x){
    int old=0;
    if (threadIdx.x==0)
      idxs[i*m+0]=old;
    for (int j=threadIdx.x;j<n;j+=blockDim.x){
      temp[blockIdx.x*n+j]=1e38;
    }
    for (int j=threadIdx.x;j<min(BufferSize,n)*3;j+=blockDim.x){
      buf[j]=dataset[i*n*3+j];
    }
    __syncthreads();
    for (int j=1;j<m;j++){
      int besti=0;
      float best=-1;
      float x1=dataset[i*n*3+old*3+0];
      float y1=dataset[i*n*3+old*3+1];
      float z1=dataset[i*n*3+old*3+2];
      for (int k=threadIdx.x;k<n;k+=blockDim.x){
        float td=temp[blockIdx.x*n+k];
        float x2,y2,z2;
        if (k<BufferSize){
          x2=buf[k*3+0];
          y2=buf[k*3+1];
          z2=buf[k*3+2];
        }else{
          x2=dataset[i*n*3+k*3+0];
          y2=dataset[i*n*3+k*3+1];
          z2=dataset[i*n*3+k*3+2];
        }
        float d=(x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1);
        float d2=min(d,td);
        if (d2!=td)
          temp[blockIdx.x*n+k]=d2;
        if (d2>best){
          best=d2;
          besti=k;
        }
      }
      dists[threadIdx.x]=best;
      dists_i[threadIdx.x]=besti;
      for (int u=0;(1<<u)<blockDim.x;u++){
        __syncthreads();
        if (threadIdx.x<(blockDim.x>>(u+1))){
          int i1=(threadIdx.x*2)<<u;
          int i2=(threadIdx.x*2+1)<<u;
          if (dists[i1]<dists[i2]){
            dists[i1]=dists[i2];
            dists_i[i1]=dists_i[i2];
          }
        }
      }
      __syncthreads();
      old=dists_i[0];
      if (threadIdx.x==0)
        idxs[i*m+j]=old;
    }
  }
}

void sample(int b, int n,const float *inp, float *result)
{
	sampleKernel<<<b,512>>>(b,n, inp,result);
}
void cubeSelectLauncher(int b, int n, float radius, const float* xyz, int* idx_out) {
    cube_select<<<b, 512>>>(b, n, radius, xyz, idx_out);
}

void groupPointLauncher(int b, int n, int c, int m, int nsample, const float *points, const int *idx, float *out){
    group_point_gpu<<<b,256>>>(b,n,c,m,nsample,points,idx,out);
    //cudaDeviceSynchronize();
}
void queryBallPointLauncher(int b, int n, float radius, int nsample, const float *xyz1, int *idx) {
    query_ball_point_gpu<<<b,512>>>(b,n,radius,nsample,xyz1,idx);
    //cudaDeviceSynchronize();
}
void knn(int b, int n,const float *xyz, int *idx_out)  
{
    knnKernel<<<b,512>>>(b,n,xyz,idx_out);
}
void gather_pointLauncher(int b, int n, int nsamples, const int* idx, const float* xyz, float* result)
{
    gather_point<<<b,512>>>(b,n,nsamples,idx,xyz,result);
}
void farthestpointsamplingLauncher(int b,int n,int m,const float * inp,float * temp,int * out){
  farthestpointsamplingKernel<<<32,512>>>(b,n,m,inp,temp,out);
}

void gatherpointLauncher(int b,int n,int m,const float * inp,const int * idx,float * out){
  gatherpointKernel<<<dim3(2,8,1),512>>>(b,n,m,inp,idx,out);
}