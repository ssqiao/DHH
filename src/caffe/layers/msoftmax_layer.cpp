#include <algorithm>
#include <vector>
#include <fstream>
#include "caffe/layers/msoftmax_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void MSoftmaxLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->ReshapeLike(*bottom[0]);
  int max_video_size = 500;
  sum_multiplier_.Reshape(max_video_size,1,1,1);
  Dtype* multiplier_data = sum_multiplier_.mutable_cpu_data();
  caffe_set(sum_multiplier_.count(), Dtype(1), multiplier_data);
  //outer_num_ = bottom[0]->num();
  video_num_ = bottom[1]->num();
  scale_.Reshape(1,1,1,1);
}

template <typename Dtype>
void MSoftmaxLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  Dtype* scale_data = scale_.mutable_cpu_data();
  caffe_copy(bottom[0]->count(), bottom_data, top_data);
  // We need to subtract the max to avoid numerical issues, compute the exp,
  // and then normalize.
  for (int i = 0; i < video_num_; ++i) {
    // initialize scale_data to the first plane
	
	int start = bottom[1]->cpu_data()[i*2];
	int end = bottom[1]->cpu_data()[i*2+1];
	int clip_size = end - start + 1;
	scale_data[0] = bottom_data[start];
    for (int j = start+1; j <= end; j++) {
      
        scale_data[0] = std::max(scale_data[0],
            bottom_data[j]);

    }
    // subtraction
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, clip_size, 1,
        1, -1., sum_multiplier_.cpu_data(), scale_data, 1., top_data);
	  
    // exponentiation
    caffe_exp<Dtype>(clip_size, top_data, top_data);
    // sum after exp
    caffe_cpu_gemv<Dtype>(CblasTrans, clip_size, 1, 1.,
        top_data, sum_multiplier_.cpu_data(), 0., scale_data);
    // division   
    for (int j = 0; j < clip_size; j++){
		caffe_div(1, top_data, scale_data, top_data);
        top_data += 1;   
	}	
    
  }
}

template <typename Dtype>
void MSoftmaxLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* top_data = top[0]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  Dtype* scale_data = scale_.mutable_cpu_data();
  caffe_copy(top[0]->count(), top_diff, bottom_diff);
  for (int i = 0; i < video_num_; ++i) {
    // compute dot(top_diff, top_data) and subtract them from the bottom diff
	int start = bottom[1]->cpu_data()[i*2];
	int end = bottom[1]->cpu_data()[i*2+1];
	int clip_size = end - start + 1;
    
    scale_data[0] = caffe_cpu_strided_dot<Dtype>(clip_size,
        bottom_diff + start, 1,
        top_data + start, 1);
    
    // subtraction
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, clip_size, 1, 1,
        -1., sum_multiplier_.cpu_data(), scale_data, 1., bottom_diff + start);
  }
  // elementwise multiplication
  caffe_mul(top[0]->count(), bottom_diff, top_data, bottom_diff);
}


#ifdef CPU_ONLY
STUB_GPU(MSoftmaxLayer);
#endif

INSTANTIATE_CLASS(MSoftmaxLayer);
REGISTER_LAYER_CLASS(MSoftmax);
}  // namespace caffe
