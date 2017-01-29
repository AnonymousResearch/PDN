#include <vector>
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/clstm_unit_layer.hpp"

namespace caffe {

template<typename Dtype>
inline Dtype sigmoid(Dtype x) {
	return 1. / (1. + exp(-x));
}

template<typename Dtype>
inline Dtype sigmoid_diff(Dtype x) {
	return x * (1. - x);
}

template<typename Dtype>
inline Dtype tanh(Dtype x) {
	return ((exp(x) - exp(-x)) / (exp(x) + exp(-x)));
	// Dtype exp2x = exp(2 * x);
	// return fabs(x) < Dtype(5) ?
	// 		((exp2x - Dtype(1)) / (exp2x + Dtype(1))) :
	// 		(x > 0 ? Dtype(1) : Dtype(-1));
}

template<typename Dtype>
inline Dtype tanh_diff(Dtype x) {
	return (1. - x * x);
}

template<typename Dtype>
void CLstmUnitLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	LstmUnitParameter lstm_unit_param = this->layer_param_.lstm_unit_param();
    TransposeParameter transpose_param = this->layer_param_.transpose_param();
	CHECK((transpose_param.has_field_size()))
			<< "transpose_param.has_field_size()";
	CHECK((lstm_unit_param.has_input_weight_filler()))
			<< "lstm_unit_param.has_input_weight_filler()";
	CHECK((lstm_unit_param.has_input_gate_weight_filler()))
			<< "lstm_unit_param.has_input_gate_weight_filler()";
	CHECK((lstm_unit_param.has_forget_gate_weight_filler()))
			<< "lstm_unit_param.has_forget_gate_weight_filler()";
	CHECK((lstm_unit_param.has_output_gate_weight_filler()))
			<< "lstm_unit_param.has_output_gate_weight_filler()";

	field_size_ = this->layer_param_.transpose_param().field_size();
	int kernel_size = 2 * field_size_ - 1; 
	full_kernel_size_ = kernel_size * kernel_size;  // 5*5 kernel -> field_size = 3
	full_channels_ = bottom[0]->shape(1);
	channels_ = bottom[1]->shape(1); // full_channels_ / full_kernel_size_;
	num_ = bottom[0]->shape(0);
	height_ = bottom[0]->shape(2);
	width_ = bottom[0]->shape(3);

	// weight initialization
	this->blobs_.resize(4);
	for (int i = 0; i < 4; ++i) {
		this->blobs_[i].reset(
				new Blob<Dtype>(full_kernel_size_, channels_, 1, 1));
	}

	shared_ptr<Filler<Dtype> > input_weight_filler(
			GetFiller<Dtype>(lstm_unit_param.input_weight_filler()));
	input_weight_filler->Fill(this->blobs_[0].get());

	shared_ptr<Filler<Dtype> > input_gate_weight_filler(
			GetFiller<Dtype>(lstm_unit_param.input_gate_weight_filler()));
	input_gate_weight_filler->Fill(this->blobs_[1].get());

	shared_ptr<Filler<Dtype> > forget_gate_weight_filler(
			GetFiller<Dtype>(lstm_unit_param.forget_gate_weight_filler()));
	forget_gate_weight_filler->Fill(this->blobs_[2].get());

	shared_ptr<Filler<Dtype> > output_gate_weight_filler(
			GetFiller<Dtype>(lstm_unit_param.output_gate_weight_filler()));
	output_gate_weight_filler->Fill(this->blobs_[3].get());
}

template<typename Dtype>
void CLstmUnitLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	CHECK(
			(this->layer_param_.bottom_size() == 2
					|| this->layer_param_.bottom_size() == 0))
			<< "LstmUnit must have a data and cell bottom";
	CHECK(
			(this->layer_param_.top_size() == 2
					|| this->layer_param_.top_size() == 0))
			<< "LstmUnit must have a data and cell top";

	CHECK(bottom[0]->count() == (bottom[1]->count() * full_kernel_size_))
				<< "The numbers must be equal. ( "
				<< bottom[0]->count() << " vs "
				<< (bottom[1]->count() * full_kernel_size_) << " ).";
	input_gates_blob_.Reshape(num_, channels_, height_, width_);
	forget_gates_blob_.Reshape(num_, channels_, height_, width_);
	output_gates_blob_.Reshape(num_, channels_, height_, width_);
	input_values_blob_.Reshape(num_, channels_, height_, width_);

	next_state_tot_diff_.Reshape(num_, channels_, height_, width_);
	state_blob_.Reshape(num_, channels_, height_, width_); 
	// input_gates_diff_.Reshape(num_, channels_, height_, width_); 
	// forget_gates_diff_.Reshape(num_, channels_, height_, width_); 
	// output_gates_diff_.Reshape(num_, channels_, height_, width_); 
	// input_values_diff_.Reshape(num_, channels_, height_, width_); 

	vector<int> shape;
	shape.push_back(num_);
	shape.push_back(channels_);
	shape.push_back(height_);
	shape.push_back(width_);
	top[0]->Reshape(shape);
	top[1]->Reshape(shape);
}

template<typename Dtype>
void CLstmUnitLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	const int spatial_dim = height_ * width_;
	// bottom data
	const Dtype* input_data = bottom[0]->cpu_data();
	const Dtype* prev_state_data = bottom[1]->cpu_data();

	//weights
	const Dtype* input_weight = this->blobs_[0]->cpu_data();
	const Dtype* input_gate_weight = this->blobs_[1]->cpu_data();
	const Dtype* forget_gate_weight = this->blobs_[2]->cpu_data();
	const Dtype* output_gate_weight = this->blobs_[3]->cpu_data();

	//gates
	Dtype* input_values = input_values_blob_.mutable_cpu_data();
	Dtype* input_gates = input_gates_blob_.mutable_cpu_data();
	Dtype* forget_gates = forget_gates_blob_.mutable_cpu_data();
	Dtype* output_gates = output_gates_blob_.mutable_cpu_data();

	Dtype* input_gates_diff = input_gates_blob_.mutable_cpu_diff();
	Dtype* forget_gates_diff = forget_gates_blob_.mutable_cpu_diff();
	Dtype* output_gates_diff = output_gates_blob_.mutable_cpu_diff();
	Dtype* input_values_diff = input_values_blob_.mutable_cpu_diff();
	Dtype* state_diff = state_blob_.mutable_cpu_diff();

	// top data
	Dtype* next_hidden_state = top[0]->mutable_cpu_data();
	Dtype* next_memory_state = top[1]->mutable_cpu_data();

	// intermediate gates 
	for (int n = 0; n < num_; n++) {
		for (int fks = 0; fks < full_kernel_size_; fks++){
			for (int c = 0; c < channels_; c++){
				int gate_offset = fks * channels_ + c;
				int input_data_offset = (n * full_channels_ + fks * channels_ + c) * spatial_dim;
				int buffer_offset = (n * channels_ + c) * spatial_dim;
				caffe_axpy(spatial_dim, input_weight[gate_offset], 
						input_data + input_data_offset, input_values + buffer_offset);
				caffe_axpy(spatial_dim, input_gate_weight[gate_offset], 
						input_data + input_data_offset, input_gates + buffer_offset);
				caffe_axpy(spatial_dim, forget_gate_weight[gate_offset], 
						input_data + input_data_offset, forget_gates + buffer_offset);
				caffe_axpy(spatial_dim, output_gate_weight[gate_offset], 
						input_data + input_data_offset, output_gates + buffer_offset);
			}
		}
	}
	
	// if (this->layer_param_.lstm_unit_param().tie_output_forget()) {
	// 	caffe_set(forget_gates_data_buffer_->count(), Dtype(0), forget_gates);
	// 	caffe_sub(forget_gates_data_buffer_->count(), forget_gates, output_gates, forget_gates);
	// }

	for (int idx = 0; idx < top[0]->count(); idx++){
		input_gates[idx] = sigmoid(input_gates[idx]);
		forget_gates[idx] = sigmoid(forget_gates[idx]);
		output_gates[idx] = sigmoid(output_gates[idx]);
		input_values[idx] = tanh(input_values[idx]);

		next_memory_state[idx] = prev_state_data[idx] * forget_gates[idx]
				+ input_gates[idx] * input_values[idx];
		next_hidden_state[idx] = tanh(next_memory_state[idx]) * output_gates[idx];

		// diff
		input_gates_diff[idx] = sigmoid_diff(input_gates[idx]);
		forget_gates_diff[idx] = sigmoid_diff(forget_gates[idx]);
		output_gates_diff[idx] = sigmoid_diff(output_gates[idx]);
		input_values_diff[idx] = tanh_diff(input_values[idx]);
		state_diff[idx] = tanh_diff(tanh(next_memory_state[idx]));
	}
}

template<typename Dtype>
void CLstmUnitLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {
	const int spatial_dim = height_ * width_;
	for (int i = 0; i < 2; ++i) {
		caffe_set(bottom[i]->count(), Dtype(0), bottom[i]->mutable_cpu_diff());
	}
	for (int i = 0; i < 4; ++i) {
		caffe_set(this->blobs_[i]->count(), Dtype(0),
				this->blobs_[i]->mutable_cpu_diff());
	}

	// weights
	const Dtype* input_weight = this->blobs_[0]->cpu_data();
	const Dtype* input_gate_weight = this->blobs_[1]->cpu_data();
	const Dtype* forget_gate_weight = this->blobs_[2]->cpu_data();
	const Dtype* output_gate_weight = this->blobs_[3]->cpu_data();

	Dtype* input_weight_diff = this->blobs_[0]->mutable_cpu_diff();
	Dtype* input_gate_weight_diff = this->blobs_[1]->mutable_cpu_diff();
	Dtype* forget_gate_weight_diff = this->blobs_[2]->mutable_cpu_diff();
	Dtype* output_gate_weight_diff = this->blobs_[3]->mutable_cpu_diff();


	// data
	const Dtype* input_data = bottom[0]->cpu_data();
	const Dtype* prev_state_data = bottom[1]->cpu_data();
	const Dtype* next_memory_state = top[1]->cpu_data();
	const Dtype* next_hidden_state_diff = top[0]->cpu_diff();
	const Dtype* next_memory_state_diff = top[1]->cpu_diff();
	Dtype* input_diff = bottom[0]->mutable_cpu_diff();
	Dtype* prev_state_diff = bottom[1]->mutable_cpu_diff();

	// gates
	const Dtype* input_values = input_values_blob_.cpu_data();
	const Dtype* input_gates = input_gates_blob_.cpu_data();
	const Dtype* forget_gates = forget_gates_blob_.cpu_data();
	const Dtype* output_gates = output_gates_blob_.cpu_data();

	const Dtype* input_gates_diff = input_gates_blob_.cpu_diff();
	const Dtype* forget_gates_diff = forget_gates_blob_.cpu_diff();
	const Dtype* output_gates_diff = output_gates_blob_.cpu_diff();
	const Dtype* input_values_diff = input_values_blob_.cpu_diff();
	const Dtype* state_diff = state_blob_.cpu_diff();

	// // diff_buffer
	// Dtype* input_values_data_m = input_values_diff_.mutable_cpu_data();
	// Dtype* input_gates_data_m = input_gates_diff_.mutable_cpu_data();
	// Dtype* forget_gates_data_m = forget_gates_diff_.mutable_cpu_data();
	// Dtype* output_gates_data_m = output_gates_diff_.mutable_cpu_data();
	// const Dtype* input_values_data = input_values_diff_.cpu_data();
	// const Dtype* input_gates_data = input_gates_diff_.cpu_data();
	// const Dtype* forget_gates_data = forget_gates_diff_.cpu_data();
	// const Dtype* output_gates_data = output_gates_diff_.cpu_data();
	// caffe_set(input_values_diff_.count(), Dtype(0), input_values_data_m);
	// caffe_set(input_gates_diff_.count(), Dtype(0), input_gates_data_m);
	// caffe_set(forget_gates_diff_.count(), Dtype(0), forget_gates_data_m);
	// caffe_set(output_gates_diff_.count(), Dtype(0), output_gates_data_m);


	// prev_state_diff
	const int count = bottom[1]->count();
	Dtype* next_state_tot_diff_m = next_state_tot_diff_.mutable_cpu_data();
	const Dtype* next_state_tot_diff = next_state_tot_diff_.cpu_data();
	caffe_mul(count, output_gates, next_hidden_state_diff, next_state_tot_diff_m);
	caffe_mul(count, next_state_tot_diff, state_diff, next_state_tot_diff_m);
	caffe_add(count, next_memory_state_diff, next_state_tot_diff, next_state_tot_diff_m);
	caffe_mul(count, next_state_tot_diff, forget_gates, prev_state_diff);

	for (int fks = 0; fks < full_kernel_size_; fks++){
		for (int c = 0; c < channels_; c++){
			int gate_weight_offset = fks * channels_ + c;
			for (int n = 0; n < num_; n++) {
				for (int sp = 0; sp < spatial_dim; sp++) {
					int output_data_offset = (n * channels_ + c) * spatial_dim + sp;
					int input_data_offset = ((n * full_kernel_size_ + fks) * channels_  + c) * spatial_dim + sp;
					// input_weight
					input_weight_diff[gate_weight_offset] += 
							next_state_tot_diff[output_data_offset] * 
							input_gates[output_data_offset] * 
							input_values_diff[output_data_offset] * 
							input_data[input_data_offset];
					// input_gate_weight
					input_gate_weight_diff[gate_weight_offset] += 
							next_state_tot_diff[output_data_offset] * 
							input_values[output_data_offset] * 
							input_gates_diff[output_data_offset] * 
							input_data[input_data_offset];
					// forget_gate_weight
					forget_gate_weight_diff[gate_weight_offset] += 
							next_state_tot_diff[output_data_offset] * 
							prev_state_data[output_data_offset] * 
							forget_gates_diff[output_data_offset] * 
							input_data[input_data_offset];
					// output_gate_weight
					output_gate_weight_diff[gate_weight_offset] += 
							next_hidden_state_diff[output_data_offset] * 
							tanh(next_memory_state[output_data_offset]) * 
							output_gates_diff[output_data_offset] * 
							input_data[input_data_offset];

					// input_data_diff
					input_diff[input_data_offset] = 
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
							tanh(next_memory_state[output_data_offset]) * 
							output_gates_diff[output_data_offset] * 
							output_gate_weight[gate_weight_offset];
				}
			}
		}
	}

	// // input_weight_diff & input_diff
	// caffe_mul(count, input_gates, input_values_diff, input_values_data_m);
	// caffe_mul(count, next_state_tot_diff, input_values_data, input_values_data_m);

	// // input_gate_weight_diff & input_diff
	// caffe_mul(count, input_gates_diff, input_values, input_gates_data_m);
	// caffe_mul(count, next_state_tot_diff, input_gates_data, input_gates_data_m);

	// // forget_gate_weight_diff & input_diff
	// caffe_mul(count, forget_gates_diff, prev_state_data, forget_gates_data_m);
	// caffe_mul(count, next_state_tot_diff, forget_gates_data, forget_gates_data_m);

	// // output_gate_weight_diff & input_diff
	// for (int i = 0; i < count; i++) {
	// 	output_gates_data_m[i] = tanh(next_memory_state[i]);
	// }
	// caffe_mul(count, output_gates_diff, output_gates_data, output_gates_data_m);
	// caffe_mul(count, next_hidden_state_diff, output_gates_data, output_gates_data_m);

	// for (int n = 0; n < num_; n++) {
	// 	int num_offset = n * full_channels_;
	// 	for (int fks = 0; fks < full_kernel_size_; fks++){
	// 		for (int c = 0; c < channels_; c++){
	// 			int gate_offset = num_offset + c * full_kernel_size_ + fks;
	// 			int input_data_offset = (num_offset + fks * channels_ + c) * spatial_dim;
	// 			int buffer_offset = (n * channels_ + c) * spatial_dim;

	// 			// input_diff
	// 			caffe_axpy(spatial_dim, input_weight[gate_offset], 
	// 					input_values_data + buffer_offset, input_diff + input_data_offset);
	// 			// input_weight_diff
	// 			caffe_cpu_gemv(CblasNoTrans, 1, spatial_dim, Dtype(1.), 
	// 					input_values_data + buffer_offset, 
	// 					input_data + input_data_offset, Dtype(0.), 
	// 					input_weight_diff + gate_offset);

	// 			// input_diff
	// 			caffe_axpy(spatial_dim, input_gate_weight[gate_offset], 
	// 					input_gates_data + buffer_offset, input_diff + input_data_offset);
	// 			// input_gate_weight_diff
	// 			caffe_cpu_gemv(CblasNoTrans, 1, spatial_dim, Dtype(1.), 
	// 					input_gates_data + buffer_offset, 
	// 					input_data + input_data_offset, Dtype(0.), 
	// 					input_gate_weight_diff + gate_offset);
				
	// 			// input_diff
	// 			caffe_axpy(spatial_dim, forget_gate_weight[gate_offset], 
	// 					forget_gates_data + buffer_offset, input_diff + input_data_offset);
	// 			// forget_gate_weight_diff
	// 			caffe_cpu_gemv(CblasNoTrans, 1, spatial_dim, Dtype(1.), 
	// 					forget_gates_data + buffer_offset, 
	// 					input_data + input_data_offset, Dtype(0.), 
	// 					forget_gate_weight_diff + gate_offset);
				
	// 			// input_diff
	// 			caffe_axpy(spatial_dim, output_gate_weight[gate_offset], 
	// 					output_gates_data + buffer_offset, input_diff + input_data_offset);
	// 			// output_gate_weight_diff
	// 			caffe_cpu_gemv(CblasNoTrans, 1, spatial_dim, Dtype(1.), 
	// 					output_gates_data + buffer_offset, 
	// 					input_data + input_data_offset, Dtype(0.), 
	// 					output_gate_weight_diff + gate_offset);
	// 		}
	// 	}
	// }
}

#ifdef CPU_ONLY
STUB_GPU(CLstmUnitLayer);
#endif

INSTANTIATE_CLASS(CLstmUnitLayer);
REGISTER_LAYER_CLASS(CLstmUnit);

}  // namespace caffe
