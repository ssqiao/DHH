#include <cfloat>
#include <vector>

#include "caffe/layers/sumvector_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SumvectorLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void SumvectorLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->Reshape(bottom[0]->num(),1,1,1);
}

template <typename Dtype>
void SumvectorLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
	Dtype* top_data = top[0]->mutable_cpu_data();
	const int count = bottom[0]->count()/bottom[0]->num();
	
	for(int i = 0; i < bottom[0]->num(); i++){
		int offset_bottom = bottom[0]->offset(i);
		top_data[i] = caffe_cpu_asum(count,bottom_data+offset_bottom);
	}
}

template <typename Dtype>
void SumvectorLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const int count = bottom[0]->count()/bottom[0]->num();
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  for(int i = 0; i < bottom[0]->num(); i++){
	  int offset_bottom = bottom[0]->offset(i);
	  for (int j = 0; j < count; j++)
		  bottom_diff[offset_bottom+j] = (bottom[0]->cpu_data()[offset_bottom+j]>0)? top_diff[i]:-top_diff[i];
  }
}

#ifdef CPU_ONLY
STUB_GPU(SumvectorLayer);
#endif

INSTANTIATE_CLASS(SumvectorLayer);
REGISTER_LAYER_CLASS(Sumvector);

}  // namespace caffe
