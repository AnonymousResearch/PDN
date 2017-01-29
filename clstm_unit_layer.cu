#include <vector>
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/clstm_unit_layer.hpp"

namespace caffe {

template<typename Dtype>
__device__ Dtype cuda_sigmoid(Dtype x) {
	return 1. / (1. + exp(-x));
}

template<typename Dtype>
__device__ Dtype cuda_sigmoid_diff(Dtype x) {
	return x * (1. - x);
}

template<typename Dtype>
__device__ Dtype cuda_tanh(Dtype x) {
	Dtype exp2x = exp(2 * x);
	return fabs(x) < Dtype(5) ?
			((exp2x - Dtype(1)) / (exp2x + Dtype(1))) :
			(x > 0 ? Dtype(1) : Dtype(-1));
}

template<typename Dtype>
__device__ Dtype cuda_tanh_diff(Dtype x) {
	return (1. - x * x);
}

template<typename Dtype>
__global__ void ForwardConvolution(int count, const Dtype* input_data,
		const Dtype* input_weight, const Dtype* input_gate_weight, 
		const Dtype* forget_gate_weight, const Dtype* output_gate_weight, 
		Dtype* input_values, Dtype* input_gates, Dtype* forget_gates, 
		Dtype* output_gates, int num_, int channels_, int full_kernel_size_, 
		int spatial_dim) {
	//idx = blockIdx.x * blockIdx.dim + threadIdx.x
	CUDA_KERNEL_LOOP(idx, count) 
	{
		int n = idx / channels_ / spatial_dim;
		int c = idx / spatial_dim % channels_;
		int sd = idx % spatial_dim;
		int num_offset = n * channels_ * full_kernel_size_;
		int buffer_offset = (n * channels_ + c) * spatial_dim + sd;
		input_values[buffer_offset] = 0;
		input_gates[buffer_offset] = 0;
		forget_gates[buffer_offset] = 0;
		output_gates[buffer_offset] = 0;

		for (int i = 0; i < full_kernel_size_; i++) {
			int gate_offset = i * channels_ + c;
			int input_data_offset = (num_offset + i * channels_ + c) * spatial_dim + sd;

			input_values[buffer_offset] += input_weight[gate_offset] * input_data[input_data_offset];
			input_gates[buffer_offset] += input_gate_weight[gate_offset] * input_data[input_data_offset];
			forget_gates[buffer_offset] += forget_gate_weight[gate_offset] * input_data[input_data_offset];
			output_gates[buffer_offset] += output_gate_weight[gate_offset] * input_data[input_data_offset];
		}
	}
}

template<typename Dtype>
__global__ void ForwardCombineGates(int count, const Dtype* prev_state_data,
		Dtype* input_gates, Dtype* forget_gates, Dtype* output_gates,
		Dtype* input_values, Dtype* next_memory_state,
		Dtype* next_hidden_state, Dtype* input_values_diff, 
		Dtype* input_gates_diff, Dtype* forget_gates_diff, 
		Dtype* output_gates_diff, Dtype* state_diff) {
	//idx = blockIdx.x * blockIdx.dim + threadIdx.x
	CUDA_KERNEL_LOOP(idx, count) 
	{
		input_gates[idx] = cuda_sigmoid(input_gates[idx]);
		forget_gates[idx] = cuda_sigmoid(forget_gates[idx]);
		output_gates[idx] = cuda_sigmoid(output_gates[idx]);
		input_values[idx] = cuda_tanh(input_values[idx]);

		next_memory_state[idx] = prev_state_data[idx] * forget_gates[idx]
				+ input_gates[idx] * input_values[idx];
		next_hidden_state[idx] = cuda_tanh(next_memory_state[idx]) * output_gates[idx];
			
		// diff
		input_gates_diff[idx] = cuda_sigmoid_diff(input_gates[idx]);
		forget_gates_diff[idx] = cuda_sigmoid_diff(forget_gates[idx]);
		output_gates_diff[idx] = cuda_sigmoid_diff(output_gates[idx]);
		input_values_diff[idx] = cuda_tanh_diff(input_values[idx]);
		state_diff[idx] = cuda_tanh_diff(cuda_tanh(next_memory_state[idx]));
	}
}


template<typename Dtype>
__global__ void BackwardGates(int count, const Dtype* input_data, const Dtype* prev_state_data, 
		const Dtype* next_memory_state, const Dtype* next_state_tot_diff, 
		const Dtype* next_hidden_state_diff, const Dtype* input_values, 
		const Dtype* input_gates, const Dtype* forget_gates, const Dtype* output_gates, 
		const Dtype* input_values_diff, const Dtype* input_gates_diff, 
		const Dtype* forget_gates_diff, const Dtype* output_gates_diff, 
		const Dtype* state_diff, const Dtype* input_weight, const Dtype* input_gate_weight, 
		const Dtype* forget_gate_weight, const Dtype* output_gate_weight,
		Dtype* input_weight_diff, Dtype* input_gate_weight_diff, 
		Dtype* forget_gate_weight_diff, Dtype* output_gate_weight_diff, 
		int num_, int channels_, int full_kernel_size_, int spatial_dim) {
	CUDA_KERNEL_LOOP(idx, count)
	{
		int fks = idx / channels_;
		int c = idx % channels_;

		for (int n = 0; n < num_; n++) {
			for (int sd = 0; sd < spatial_dim; sd++) {
				// int gate_weight_offset = fks * channels_ + c;
				int output_data_offset = (n * channels_ + c) * spatial_dim + sd;
				int input_data_offset = ((n * full_kernel_size_ + fks) * channels_  + c) * spatial_dim + sd;

				// input_weight
				input_weight_diff[idx] = input_weight_diff[idx] + 
						next_state_tot_diff[output_data_offset] * 
						input_gates[output_data_offset] * 
						input_values_diff[output_data_offset] * 
						input_data[input_data_offset];
				// input_gate_weight
				input_gate_weight_diff[idx] = input_gate_weight_diff[idx] + 
						next_state_tot_diff[output_data_offset] * 
						input_values[output_data_offset] * 
						input_gates_diff[output_data_offset] * 
						input_data[input_data_offset];
				// forget_gate_weight
				forget_gate_weight_diff[idx] = forget_gate_weight_diff[idx] + 
						next_state_tot_diff[output_data_offset] * 
						prev_state_data[output_data_offset] * 
						forget_gates_diff[output_data_offset] * 
						input_data[input_data_offset];
				// output_gate_weight
				output_gate_weight_diff[idx] = output_gate_weight_diff[idx] + 
						next_hidden_state_diff[output_data_offset] * 
						cuda_tanh(next_memory_state[output_data_offset]) * 
						output_gates_diff[output_data_offset] * 
						input_data[input_data_offset];
			}
		}
	}
}

template<typename Dtype>
__global__ void BackwardInput(int count, const Dtype* input_data, const Dtype* prev_state_data, 
		const Dtype* next_memory_state, const Dtype* next_state_tot_diff, 
		const Dtype* next_hidden_state_diff, const Dtype* input_values, 
		const Dtype* input_gates, const Dtype* forget_gates, const Dtype* output_gates, 
		const Dtype* input_values_diff, const Dtype* input_gates_diff, 
		const Dtype* forget_gates_diff, const Dtype* output_gates_diff, 
		const Dtype* state_diff, const Dtype* input_weight, const Dtype* input_gate_weight, 
		const Dtype* forget_gate_weight, const Dtype* output_gate_weight, 
		Dtype* input_diff, int num_, int channels_, int full_kernel_size_, 
		int spatial_dim) {
	CUDA_KERNEL_LOOP(idx, count)
	{
		int n = idx / full_kernel_size_ / channels_ / spatial_dim;
		int fks = idx / channels_ / spatial_dim % full_kernel_size_;
		int c = idx / spatial_dim % channels_;
		int sd = idx % spatial_dim;
		int gate_weight_offset = fks * channels_ + c;
		int output_data_offset = (n * channels_ + c) * spatial_dim + sd;
		// int input_data_offset = ((n * full_kernel_size_ + fks) * channels_  + c) * spatial_dim + sd;

		// input_data_diff
		input_diff[idx] = 
				next_state_tot_diff[output_data_offset] * 
				input_gates[output_data_offset] * 
				input_values_diff[output_data_offset] * 
				input_weight[gate_weight_offset]
				+
				next_state_tot_diff[output_data_offset] * 
				input_values[output_data_offset] * 
				input_gates_diff[output_data_offset] * 
				input_gate_weight[gate_weight_offset]
				+
				next_state_tot_diff[output_data_offset] * 
				prev_state_data[output_data_offset] * 
				forget_gates_diff[output_data_offset] * 
				forget_gate_weight[gate_weight_offset]
				+
				next_hidden_state_diff[output_data_offset] * 
				cuda_tanh(next_memory_state[output_data_offset]) * 
				output_gates_diff[output_data_offset] * 
				output_gate_weight[gate_weight_offset];
	}
}

template<typename Dtype>
void CLstmUnitLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	// Forward_cpu(bottom, top);
	// bottom data
	const Dtype* input_data = bottom[0]->gpu_data();
	const Dtype* prev_state_data = bottom[1]->gpu_data();
	// weights
	const Dtype* input_weight = this->blobs_[0]->gpu_data();
	const Dtype* input_gate_weight = this->blobs_[1]->gpu_data();
	const Dtype* forget_gate_weight = this->blobs_[2]->gpu_data();
	const Dtype* output_gate_weight = this->blobs_[3]->gpu_data();

	// top data
	Dtype* next_hidden_state = top[0]->mutable_gpu_data();
	Dtype* next_memory_state = top[1]->mutable_gpu_data();

	// intermediategates
	Dtype* input_gates = input_gates_blob_.mutable_gpu_data();
	Dtype* forget_gates = forget_gates_blob_.mutable_gpu_data();
	Dtype* output_gates = output_gates_blob_.mutable_gpu_data();
	Dtype* input_values = input_values_blob_.mutable_gpu_data();

	Dtype* input_gates_diff = input_gates_blob_.mutable_gpu_diff();
	Dtype* forget_gates_diff = forget_gates_blob_.mutable_gpu_diff();
	Dtype* output_gates_diff = output_gates_blob_.mutable_gpu_diff();
	Dtype* input_values_diff = input_values_blob_.mutable_gpu_diff();
	Dtype* state_diff = state_blob_.mutable_gpu_diff();

	const int spatial_dim = height_ * width_;
	int count = num_ * channels_ * spatial_dim;
	ForwardConvolution<Dtype> <<<CAFFE_GET_BLOCKS(count),
			CAFFE_CUDA_NUM_THREADS>>>(count, input_data, input_weight, 
			input_gate_weight, forget_gate_weight, output_gate_weight, 
			input_values, input_gates, forget_gates, output_gates, 
			num_, channels_, full_kernel_size_, spatial_dim);

	// NOLINT_NEXT_LINE(whitespace/operators)
	ForwardCombineGates<Dtype> <<<CAFFE_GET_BLOCKS(count),
			CAFFE_CUDA_NUM_THREADS>>>(count, prev_state_data, input_gates,
			forget_gates, output_gates, input_values, next_memory_state,
			next_hidden_state, input_values_diff, input_gates_diff, 
			forget_gates_diff, output_gates_diff, state_diff);
	CUDA_POST_KERNEL_CHECK;
}

template<typename Dtype>
void CLstmUnitLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {
	// Backward_cpu(top, propagate_down, bottom);
	for (int i = 0; i < 2; ++i) {
		caffe_gpu_set(bottom[i]->count(), Dtype(0),
				bottom[i]->mutable_gpu_diff());
	}
	for (int i = 0; i < 4; ++i) {
		caffe_gpu_set(this->blobs_[i]->count(), Dtype(0),
				this->blobs_[i]->mutable_gpu_diff());
	}


	// weights
	const Dtype* input_weight = this->blobs_[0]->gpu_data();
	const Dtype* input_gate_weight = this->blobs_[1]->gpu_data();
	const Dtype* forget_gate_weight = this->blobs_[2]->gpu_data();
	const Dtype* output_gate_weight = this->blobs_[3]->gpu_data();

	Dtype* input_weight_diff = this->blobs_[0]->mutable_gpu_diff();
	Dtype* input_gate_weight_diff = this->blobs_[1]->mutable_gpu_diff();
	Dtype* forget_gate_weight_diff = this->blobs_[2]->mutable_gpu_diff();
	Dtype* output_gate_weight_diff = this->blobs_[3]->mutable_gpu_diff();


	// data
	const Dtype* input_data = bottom[0]->gpu_data();
	const Dtype* prev_state_data = bottom[1]->gpu_data();
	const Dtype* next_memory_state = top[1]->gpu_data();
	const Dtype* next_hidden_state_diff = top[0]->gpu_diff();
	const Dtype* next_memory_state_diff = top[1]->gpu_diff();
	Dtype* input_diff = bottom[0]->mutable_gpu_diff();
	Dtype* prev_state_diff = bottom[1]->mutable_gpu_diff();

	// gates
	const Dtype* input_values = input_values_blob_.gpu_data();
	const Dtype* input_gates = input_gates_blob_.gpu_data();
	const Dtype* forget_gates = forget_gates_blob_.gpu_data();
	const Dtype* output_gates = output_gates_blob_.gpu_data();

	const Dtype* input_gates_diff = input_gates_blob_.gpu_diff();
	const Dtype* forget_gates_diff = forget_gates_blob_.gpu_diff();
	const Dtype* output_gates_diff = output_gates_blob_.gpu_diff();
	const Dtype* input_values_diff = input_values_blob_.gpu_diff();
	const Dtype* state_diff = state_blob_.gpu_diff();


	// prev_state_diff
	const int count1 = bottom[1]->count();
	Dtype* next_state_tot_diff_m = next_state_tot_diff_.mutable_gpu_data();
	const Dtype* next_state_tot_diff = next_state_tot_diff_.gpu_data();
	caffe_gpu_mul(count1, output_gates, next_hidden_state_diff, next_state_tot_diff_m);
	caffe_gpu_mul(count1, next_state_tot_diff, state_diff, next_state_tot_diff_m);
	caffe_gpu_add(count1, next_memory_state_diff, next_state_tot_diff, next_state_tot_diff_m);
	caffe_gpu_mul(count1, next_state_tot_diff, forget_gates, prev_state_diff);

	const int spatial_dim = height_ * width_;
	const int count_gate = channels_ * full_kernel_size_;
	BackwardGates<Dtype> <<<CAFFE_GET_BLOCKS(count_gate), CAFFE_CUDA_NUM_THREADS>>>(
			count_gate, input_data, prev_state_data, next_memory_state, 
			next_state_tot_diff, next_hidden_state_diff, 
			input_values, input_gates, forget_gates, output_gates, 
			input_values_diff, input_gates_diff, forget_gates_diff, 
			output_gates_diff, state_diff, input_weight, input_gate_weight, 
			forget_gate_weight, output_gate_weight,
			input_weight_diff, input_gate_weight_diff, forget_gate_weight_diff, 
			output_gate_weight_diff, 
			num_, channels_, full_kernel_size_, spatial_dim);

	const int count_input = bottom[0]->count();
	BackwardInput<Dtype> <<<CAFFE_GET_BLOCKS(count_input), CAFFE_CUDA_NUM_THREADS>>>(
			count_input, input_data, prev_state_data, next_memory_state, 
			next_state_tot_diff, next_hidden_state_diff, 
			input_values, input_gates, forget_gates, output_gates, 
			input_values_diff, input_gates_diff, forget_gates_diff, 
			output_gates_diff, state_diff, input_weight, input_gate_weight, 
			forget_gate_weight, output_gate_weight, input_diff, 
			num_, channels_, full_kernel_size_, spatial_dim);
	CUDA_POST_KERNEL_CHECK;

}

INSTANTIATE_LAYER_GPU_FUNCS(CLstmUnitLayer);

}  // namespace caffe
