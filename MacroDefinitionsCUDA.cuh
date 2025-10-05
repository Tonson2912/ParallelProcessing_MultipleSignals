#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include<cufft.h>

#define CHECK_CUDA(errorInfo,cudaSuccInfo){ \
	if((errorInfo)!=cudaSuccInfo){  \
		fprintf(stderr,"CUDA error in line %d of file %s\:%s\n",__LINE__, __FILE__, cudaGetErrorString(cudaGetLastError()));\
		system("pause");\
		exit(-1);\
	}\
} \

//DEBUG编译条件下输出CUDA错误信息
#if _DEBUG
#define RUNTIME_CUDA_ERROR(errorInfo) CHECK_CUDA(errorInfo,cudaSuccess)		   //runtime API函数
#define CUFFT_CUDA_ERROR(errorInfo) CHECK_CUDA(errorInfo,CUFFT_SUCCESS)        //cuBlas API函数
#else
#define RUNTIME_CUDA_ERROR(errorInfo) errorInfo
#define CUFFT_CUDA_ERROR(errorInfo) errorInfo
#endif

//CUDA释放宏
#define CUDA_FREE(d_ptr){ \
	if (d_ptr != nullptr)RUNTIME_CUDA_ERROR(cudaFree(d_ptr)); d_ptr = nullptr;\
}\

//读GPU数据宏
#define READ_CUDA_DATA(d_ptr,type,start_index,length){\
	type* check=new type[length];\
	cudaMemcpy(check,d_ptr+start_index,sizeof(type)*length,cudaMemcpyDeviceToHost);\
	for(int i=start_index;i<start_index+length;++i){\
		cout << i + 1 << "  :" << check[i-start_index] << endl;\
	}\
}\
//读GPU复数数据宏
#define READ_CUDA_DATA_COMP(d_ptr,start_index,length){\
	cufftDoubleComplex* check=new cufftDoubleComplex[length];\
	cudaMemcpy(check,d_ptr+start_index,sizeof(cufftDoubleComplex)*length,cudaMemcpyDeviceToHost);\
	for(int i=start_index;i<start_index+length;++i){\
		cout << i + 1 << "  x:" << check[i-start_index].x<<" y:"<<check[i-start_index].y << endl;\
	}\
}\
