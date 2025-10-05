#pragma once
/*************************************************************************
	> ģ��:
	> �˰汾�����龰:
	> �����ⲿ��:
	> ʱ��:
	> �ļ���
 ************************************************************************/
#define _CRT_SECURE_NO_WARNINGS
#include<iostream>

using namespace std;

enum FilterMode
{
	//LOWPASS,
	//HiGHPASS,
	BANDPASS,//����ͨ�˲�
	NON_BANDPASS,//����ͨ�˲�
};

enum ReconstructionAlgorithm
{
	EXTREME_VALUE,//��ֵ��
	CENTER_OF_GRAVITY,//���ķ�
	IMPROVED_CENTER_OF_GRAVITY,//�ݶ����ķ�
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

//ģ�͹�������
reconstruction_algorithm_model<float>* reconstruction3D_GPU(unsigned int height, unsigned int width, unsigned int num, unsigned int patch_num = 1, unsigned int stream_num = 1);
signal_process_model<float>* signal_process_GPU(unsigned int height, unsigned int width, unsigned int num, unsigned int patch_num = 1, unsigned int stream_num = 1);
cuda_memery_manager<float>* memery_manager_GPU();
