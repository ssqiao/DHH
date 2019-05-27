#include <algorithm>
#include <vector>

#include "caffe/layers/quantization_loss_layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void QuantizationLossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  CHECK_EQ(bottom[0]->height(), 1);//check every dim's size, bottom[0]->data, bottom[1]->label
  CHECK_EQ(bottom[0]->width(), 1);
  high_ = this->layer_param_.quantization_loss_param().high();
  low_ = this->layer_param_.quantization_loss_param().low();
  threshold_ = (high_ + low_)/2.0;
}

template <typename Dtype>
void QuantizationLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // initialize parameters
  Dtype* bout = bottom[0]->mutable_cpu_diff();//store the gradient of the loss with respect to  every sample in previous layer's output, i.e loss layer's input
  const int num = bottom[0]->num();//get the num of samples
  const Dtype alpha = top[0]->cpu_diff()[0] / static_cast<Dtype>(bottom[0]->num());//loss term's normlization's size
  const int channels = bottom[0]->channels();//the dim of each input data (output of last full connect layer)
  Dtype loss(0.0);//calc loss value, tmp
  Dtype data(0.0);//get a dim value from each input data
  caffe_set(channels*num, Dtype(0.0), bout);//init the bout as all zero channels*num matrix
  // calculate loss and gradient, generate pair online
  for (int i = 0; i < num; ++i) {
    for (int k = 0; k < channels;k++){
      data = *(bottom[0]->cpu_data()+(i*channels)+k);
      // gradient corresponding to the regularizer
	  if (data>=threshold_){
		  *(bout + (i*channels) + k) += (data>=high_) ? (alpha):(-alpha);
		  data = (data>=high_) ? (data-high_):(high_-data); 
	  }
	  else{
		  *(bout + (i*channels) + k) += (data>=low_) ? (alpha):(-alpha);
		  data = (data>=low_) ? (data-low_):(low_-data);
	  }
      loss += alpha * data;
	  //loss += alpha * (data*data);
      
    }
  }
  top[0]->mutable_cpu_data()[0] = loss;// pass loss to loss layer's output
}

template <typename Dtype>// havn't used it, calc the loss and gradient at the same time in forward_cpu func
void QuantizationLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
   }
}

#ifdef CPU_ONLY
STUB_GPU(QuantizationLossLayer);
#endif

INSTANTIATE_CLASS(QuantizationLossLayer);
REGISTER_LAYER_CLASS(QuantizationLoss);

}  // namespace caffe
