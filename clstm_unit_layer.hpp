#ifndef CAFFE_CLSTM_UNIT_LAYER_HPP
#define CAFFE_CLSTM_UNIT_LAYER_HPP

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/proto/caffe.pb.h"


namespace caffe {

template<typename Dtype>
class CLstmUnitLayer: public Layer<Dtype> {
public:
	explicit CLstmUnitLayer(const LayerParameter& param) :
			Layer<Dtype>(param) {
	}
	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
	virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

	virtual inline bool overwrites_param_diffs() {
		return true;
	}
	virtual inline const char* type() const {
		return "CLstmUnit";
	}

protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down,
			const vector<Blob<Dtype>*>& bottom);
	virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down,
			const vector<Blob<Dtype>*>& bottom);

	int field_size_;
	int full_kernel_size_;
	int channels_;
	int full_channels_;
	// int all_mem_cells_dimension;  // all_mem_cells_dimension;
	// int all_mem_cells_num;  // all_mem_cells_num;
	int num_;  // batch size;
	// int input_data_size_;
	int height_;
	int width_;
	int M_;
	int N_;
	int K_;
	Blob<Dtype> input_gates_blob_;
	Blob<Dtype> forget_gates_blob_;
	Blob<Dtype> output_gates_blob_;
	Blob<Dtype> input_values_blob_;
	Blob<Dtype> state_blob_;

	Blob<Dtype> next_state_tot_diff_;
// 	Blob<Dtype> input_gates_diff_;
// 	Blob<Dtype> forget_gates_diff_;
// 	Blob<Dtype> output_gates_diff_;
// 	Blob<Dtype> input_values_diff_;
};

}  // namespace caffe

#endif  // CAFFE_CLSTM_UNIT_LAYER_HPP
