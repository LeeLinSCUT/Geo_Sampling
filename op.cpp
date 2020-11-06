#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include <cuda_runtime.h>

using namespace tensorflow;

// input: radius (1), nsample (1), xyz1 (b,n,3)
// output: idx (b,n,nsample) 
REGISTER_OP("QueryBallPoint")
    .Attr("radius: float")
    .Attr("nsample: int")
    .Input("xyz1: float32")
    .Output("idx: int32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        ::tensorflow::shape_inference::ShapeHandle dims1; // batch_size * npoint * 3
        c->WithRank(c->input(0), 3, &dims1);
        int nsample;
        TF_RETURN_IF_ERROR(c->GetAttr("nsample", &nsample));
        ::tensorflow::shape_inference::ShapeHandle output1 = c->MakeShape({c->Dim(dims1, 0), c->Dim(dims1, 1), nsample});
        c->set_output(0, output1);
        return Status::OK();
    });

REGISTER_OP("Sampledemo")   //定义Op的接口
  .Input("inp: float32") //c:(1,2048,8,3) result:(1,2048)
  .Output("out: float32")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
  	        ::tensorflow::shape_inference::ShapeHandle dim1;
            c->WithRank(c->input(0), 4, &dim1); // batch_size * npoint * 3
            ::tensorflow::shape_inference::ShapeHandle output1 = c->MakeShape({c->Dim(dim1, 0), c->Dim(dim1, 1)});
            c->set_output(0, output1); // batch_size * npoint 
            return Status::OK();
  });

REGISTER_OP("Knndemo")   //定义Op的接口
  .Input("xyz: float32") //(b,n,3)
  .Output("idx: int32") //(b,n,6)
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
  	        ::tensorflow::shape_inference::ShapeHandle dim1;
            c->WithRank(c->input(0), 3, &dim1); // batch_size * npoint * 3
            ::tensorflow::shape_inference::ShapeHandle output1 = c->MakeShape({c->Dim(dim1, 0), c->Dim(dim1, 1), 6});
            c->set_output(0, output1); // batch_size * npoint * 6 
            return Status::OK();
  });
REGISTER_OP("GatherPoint")   //定义Op的接口
  .Input("xyz: float32") //(b,n,3)
  .Input("idx: int32")   //(b,nsamples)
  .Output("out: float32") //(b,nsamples,3)
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
            ::tensorflow::shape_inference::ShapeHandle dims1; 
        	c->WithRank(c->input(1), 2, &dims1); //(b,nsamples)
            ::tensorflow::shape_inference::ShapeHandle output1 = c->MakeShape({c->Dim(dims1, 0), c->Dim(dims1, 1), 3});
            c->set_output(0, output1); // batch_size * npoint * 3 
            return Status::OK();
  });
REGISTER_OP("GatherPointfar")
  .Input("inp: float32")
  .Input("idx: int32")
  .Output("out: float32")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    ::tensorflow::shape_inference::ShapeHandle dims1; // batch_size * ndataset * 3
    c->WithRank(c->input(0), 3, &dims1);
    ::tensorflow::shape_inference::ShapeHandle dims2; // batch_size * npoints
    c->WithRank(c->input(1), 2, &dims2);
    // batch_size * npoints * 3
    ::tensorflow::shape_inference::ShapeHandle output = c->MakeShape({c->Dim(dims1, 0), c->Dim(dims2, 1), c->Dim(dims1, 2)});
    c->set_output(0, output);
    return Status::OK();
  });
REGISTER_OP("CubeSelect")
        .Attr("radius: float")
        .Input("xyz: float32")
        .Output("idx: int32")
        .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
            ::tensorflow::shape_inference::ShapeHandle dim1;
            c->WithRank(c->input(0), 3, &dim1); // batch_size * npoint * 3
            ::tensorflow::shape_inference::ShapeHandle output1 = c->MakeShape({c->Dim(dim1, 0), c->Dim(dim1, 1), 8});
            c->set_output(0, output1); // batch_size * npoint * 8
            return Status::OK();
        });
REGISTER_OP("GroupPoint")
    .Input("points: float32")
    .Input("idx: int32")
    .Output("out: float32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        ::tensorflow::shape_inference::ShapeHandle dims1; // batch_size * ndataset * channels
        c->WithRank(c->input(0), 3, &dims1);
        ::tensorflow::shape_inference::ShapeHandle dims2; // batch_size * npoints * nsample
        c->WithRank(c->input(1), 3, &dims2);
        // batch_size * npoints * nsample * channels
        ::tensorflow::shape_inference::ShapeHandle output = c->MakeShape({c->Dim(dims2, 0), c->Dim(dims2, 1), c->Dim(dims2, 2), c->Dim(dims1, 2)});
        c->set_output(0, output);
        return Status::OK();
    });
REGISTER_OP("FarthestPointSample")
  .Attr("npoint: int")
  .Input("inp: float32")
  .Output("out: int32")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    ::tensorflow::shape_inference::ShapeHandle dims1; // batch_size * npoint * 3
    c->WithRank(c->input(0), 3, &dims1);
    int npoint;
    TF_RETURN_IF_ERROR(c->GetAttr("npoint", &npoint));
    ::tensorflow::shape_inference::ShapeHandle output = c->MakeShape({c->Dim(dims1, 0), npoint});
    c->set_output(0, output);
    return Status::OK();
  });
void sample(int b, int n, const float *c, float *result);
class sampleGpuOp: public OpKernel{
  public:
    explicit sampleGpuOp(OpKernelConstruction* context):OpKernel(context) {   //OpKernelContext是作为OpKernel的核心API Compute函数的参数，所有计算相关的参数都会包含在这个对象中。
    }
    void Compute(OpKernelContext * context)override{
      const Tensor& inp_tensor=context->input(0); //inp_tensor(2048,3) c:(1,2048,8,3) result:(1,2048)
      int b=inp_tensor.shape().dim_size(0);  //1
      int n=inp_tensor.shape().dim_size(1);  //2048
      auto inp_flat=inp_tensor.flat<float>();
      const float * inp=&(inp_flat(0)); 

      Tensor * out_tensor;
      OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape{b, n},&out_tensor)); //这里申请输出变量output (1,2048)
      auto out_flat=out_tensor->flat<float>();
      float * out=&(out_flat(0));
      sample(b,n,inp,out);  //inp:2048*8*3  out:(1,2048)
    }
};
REGISTER_KERNEL_BUILDER(Name("Sampledemo").Device(DEVICE_GPU), sampleGpuOp);

void knn(int b, int n,const float *xyz, int *idx_out);  
class knnGpuOp: public OpKernel{
  public:
    explicit knnGpuOp(OpKernelConstruction* context):OpKernel(context) {   //OpKernelContext是作为OpKernel的核心API Compute函数的参数，所有计算相关的参数都会包含在这个对象中。
    }
    void Compute(OpKernelContext * context)override{
      const Tensor& inp_tensor=context->input(0); //inp_tensor(1,2048,3) 
      int b=inp_tensor.shape().dim_size(0);  //1
      int n=inp_tensor.shape().dim_size(1);  //2048
      auto inp_flat=inp_tensor.flat<float>();
      const float * inp=&(inp_flat(0)); 

      Tensor * out_tensor;
      OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape{b, n, 6},&out_tensor)); //这里申请输出变量output (1,2048)
      auto out_flat=out_tensor->flat<int>();
      int * out=&(out_flat(0));
      knn(b,n,inp,out);  //inp:2048*8*3  out:(1,2048)
    }
};
REGISTER_KERNEL_BUILDER(Name("Knndemo").Device(DEVICE_GPU), knnGpuOp);

void cubeSelectLauncher(int b, int n, float radius, const float* xyz, int* idx_out);
class CubeSelectOp : public OpKernel {
public:
    explicit CubeSelectOp(OpKernelConstruction * context): OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("radius", &radius_));
    }

    void Compute(OpKernelContext* context) override {
        const Tensor& xyz_tensor = context->input(0);
        int b = xyz_tensor.shape().dim_size(0);
        int n = xyz_tensor.shape().dim_size(1);

        Tensor* idx_out_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{b, n, 8}, &idx_out_tensor));
        auto xyz_flat = xyz_tensor.flat<float>();
        const float* xyz = &(xyz_flat(0));

        auto idx_out_flat = idx_out_tensor->flat<int>();
        int* idx_out = &(idx_out_flat(0));
        cubeSelectLauncher(b, n, radius_, xyz, idx_out);
    }
private:
    float radius_;
};
REGISTER_KERNEL_BUILDER(Name("CubeSelect").Device(DEVICE_GPU), CubeSelectOp);

void groupPointLauncher(int b, int n, int c, int m, int nsample, const float *points, const int *idx, float *out);
class GroupPointGpuOp: public OpKernel{
    public:
        explicit GroupPointGpuOp(OpKernelConstruction * context):OpKernel(context){}

        void Compute(OpKernelContext * context) override {
            const Tensor& points_tensor=context->input(0);
            OP_REQUIRES(context, points_tensor.dims()==3, errors::InvalidArgument("GroupPoint expects (batch_size, num_points, channel) points shape"));
            int b = points_tensor.shape().dim_size(0);
            int n = points_tensor.shape().dim_size(1);
            int c = points_tensor.shape().dim_size(2);

            const Tensor& idx_tensor=context->input(1);
            OP_REQUIRES(context,idx_tensor.dims()==3 && idx_tensor.shape().dim_size(0)==b, errors::InvalidArgument("GroupPoint expects (batch_size, npoints, nsample) idx shape"));
            int m = idx_tensor.shape().dim_size(1);
            int nsample = idx_tensor.shape().dim_size(2);

            Tensor * out_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(0,TensorShape{b,m,nsample,c}, &out_tensor));

            auto points_flat = points_tensor.flat<float>();
            const float *points = &(points_flat(0));
            auto idx_flat = idx_tensor.flat<int>();
            const int *idx = &(idx_flat(0));
            auto out_flat = out_tensor->flat<float>();
            float *out = &(out_flat(0));
            groupPointLauncher(b,n,c,m,nsample,points,idx,out);
        }
};
REGISTER_KERNEL_BUILDER(Name("GroupPoint").Device(DEVICE_GPU),GroupPointGpuOp);

void queryBallPointLauncher(int b, int n, float radius, int nsample, const float *xyz1, int *idx);
class QueryBallPointGpuOp : public OpKernel {
    public:
        explicit QueryBallPointGpuOp(OpKernelConstruction* context) : OpKernel(context) {
            OP_REQUIRES_OK(context, context->GetAttr("radius", &radius_));
            OP_REQUIRES_OK(context, context->GetAttr("nsample", &nsample_));
        }

        void Compute(OpKernelContext* context) override {
            const Tensor& xyz1_tensor = context->input(0);
            int b = xyz1_tensor.shape().dim_size(0);
            int n = xyz1_tensor.shape().dim_size(1);

            Tensor *idx_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{b,n,nsample_}, &idx_tensor));

            auto xyz1_flat = xyz1_tensor.flat<float>();
            const float *xyz1 = &(xyz1_flat(0));

            auto idx_flat = idx_tensor->flat<int>();
            int *idx = &(idx_flat(0));

            queryBallPointLauncher(b,n,radius_,nsample_,xyz1,idx);
        }
    private:
        float radius_;
        int nsample_;
};
REGISTER_KERNEL_BUILDER(Name("QueryBallPoint").Device(DEVICE_GPU), QueryBallPointGpuOp);

void gather_pointLauncher(int b, int n, int nsamples, const int* idx, const float* xyz, float* result);
class gatherpointGpuOp: public OpKernel{
  public:
    explicit gatherpointGpuOp(OpKernelConstruction* context):OpKernel(context) {   //OpKernelContext是作为OpKernel的核心API Compute函数的参数，所有计算相关的参数都会包含在这个对象中。
    }
    void Compute(OpKernelContext * context)override{
      const Tensor& inp_tensor=context->input(0); //(1,2048,3)
      int b=inp_tensor.shape().dim_size(0);  //1
      int n=inp_tensor.shape().dim_size(1);  //2048
      auto inp_flat=inp_tensor.flat<float>();
      const float * xyz=&(inp_flat(0)); 

      const Tensor& inp_tensor1=context->input(1); //(1,1024)
      int nsamples=inp_tensor1.shape().dim_size(1);  //nsamples
      auto inp_flat1=inp_tensor1.flat<int>();
      const int * idx=&(inp_flat1(0)); 

      Tensor * out_tensor;
      OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape{b, nsamples, 3},&out_tensor)); //这里申请输出变量output (b,nsamples,3)
      auto out_flat=out_tensor->flat<float>();
      float * out=&(out_flat(0));
      gather_pointLauncher(b,n,nsamples,idx,xyz,out);  //inp:2048*8*3  out:(1,2048)
    }
};
REGISTER_KERNEL_BUILDER(Name("GatherPoint").Device(DEVICE_GPU), gatherpointGpuOp);

void gatherpointLauncher(int b,int n,int m,const float * inp,const int * idx,float * out);
class GatherPointGpuOp: public OpKernel{
  public:
    explicit GatherPointGpuOp(OpKernelConstruction * context):OpKernel(context){}
    void Compute(OpKernelContext * context)override{
      const Tensor& inp_tensor=context->input(0);
      OP_REQUIRES(context,inp_tensor.dims()==3 && inp_tensor.shape().dim_size(2)==3,errors::InvalidArgument("GatherPoint expects (batch_size,num_points,3) inp shape"));
      int b=inp_tensor.shape().dim_size(0);
      int n=inp_tensor.shape().dim_size(1);
      const Tensor& idx_tensor=context->input(1);
      OP_REQUIRES(context,idx_tensor.dims()==2 && idx_tensor.shape().dim_size(0)==b,errors::InvalidArgument("GatherPoint expects (batch_size,num_result) idx shape"));
      int m=idx_tensor.shape().dim_size(1);
      auto inp_flat=inp_tensor.flat<float>();
      const float * inp=&(inp_flat(0));
      auto idx_flat=idx_tensor.flat<int>();
      const int * idx=&(idx_flat(0));
      Tensor * out_tensor=NULL;
      OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape{b,m,3},&out_tensor));
      auto out_flat=out_tensor->flat<float>();
      float * out=&(out_flat(0));
      gatherpointLauncher(b,n,m,inp,idx,out);
    }
};
REGISTER_KERNEL_BUILDER(Name("GatherPointfar").Device(DEVICE_GPU),GatherPointGpuOp);

void farthestpointsamplingLauncher(int b,int n,int m,const float * inp,float * temp,int * out);
class FarthestPointSampleGpuOp: public OpKernel{
  public:
    explicit FarthestPointSampleGpuOp(OpKernelConstruction* context):OpKernel(context) {
                    OP_REQUIRES_OK(context, context->GetAttr("npoint", &npoint_));
                    OP_REQUIRES(context, npoint_ > 0, errors::InvalidArgument("FarthestPointSample expects positive npoint"));
                }
    void Compute(OpKernelContext * context)override{
      int m = npoint_;

      const Tensor& inp_tensor=context->input(0);
      OP_REQUIRES(context,inp_tensor.dims()==3 && inp_tensor.shape().dim_size(2)==3,errors::InvalidArgument("FarthestPointSample expects (batch_size,num_points,3) inp shape"));
      int b=inp_tensor.shape().dim_size(0);
      int n=inp_tensor.shape().dim_size(1);
      auto inp_flat=inp_tensor.flat<float>();
      const float * inp=&(inp_flat(0));
      Tensor * out_tensor;
      OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape{b,m},&out_tensor));
      auto out_flat=out_tensor->flat<int>();
      int * out=&(out_flat(0));
      Tensor temp_tensor;
      OP_REQUIRES_OK(context,context->allocate_temp(DataTypeToEnum<float>::value,TensorShape{32,n},&temp_tensor));
      auto temp_flat=temp_tensor.flat<float>();
      float * temp=&(temp_flat(0));
      farthestpointsamplingLauncher(b,n,m,inp,temp,out);
    }
    private:
        int npoint_;
};
REGISTER_KERNEL_BUILDER(Name("FarthestPointSample").Device(DEVICE_GPU),FarthestPointSampleGpuOp);