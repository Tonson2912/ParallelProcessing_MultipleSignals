#pragma once
/*************************************************************************
	> 模块:
	> 此版本适用情景:
	> 链接外部库:
	> 时间:
	> 文件：
 ************************************************************************/
#define _CRT_SECURE_NO_WARNINGS
#include<iostream>

using namespace std;

enum FilterMode
{
	//LOWPASS,
	//HiGHPASS,
	BANDPASS,//带内通滤波
	NON_BANDPASS,//带外通滤波
};

enum ReconstructionAlgorithm
{
	EXTREME_VALUE,//极值法
	CENTER_OF_GRAVITY,//质心法
	IMPROVED_CENTER_OF_GRAVITY,//梯度质心法
};

template<typename T>
class cuda_memery_manager
{
public:
	virtual bool mallocMemery(void** memery_ptr, unsigned long long size) = 0;
	virtual void freeMemery(void* memery_ptr) = 0;
};

template<class Y>
class signal_process_model
{
public:
	virtual void compute_gpu(Y* datalines_src, Y* signal_real_dst, Y* signal_abs_dst,bool is_need_trans) = 0;
	virtual void set_frequency(int low_freq, int high_freq,FilterMode fliter_mode) = 0;
	virtual void del_object() = 0;
};


template<class T> 
class reconstruction_algorithm_model
{
public:
	virtual void reconstruction3D_compute_gpu(T* data3D_src, T* data2D_dst, ReconstructionAlgorithm algorithm) = 0;
	virtual void del_object() = 0;
};

//模型工厂函数
reconstruction_algorithm_model<float>* reconstruction3D_GPU(unsigned int height, unsigned int width, unsigned int num, unsigned int patch_num = 1, unsigned int stream_num = 1);
signal_process_model<float>* signal_process_GPU(unsigned int height, unsigned int width, unsigned int num, unsigned int patch_num = 1, unsigned int stream_num = 1);
cuda_memery_manager<float>* memery_manager_GPU();
