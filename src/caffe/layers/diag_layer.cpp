#include <vector>

#include "caffe/layers/diag_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void DiagLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	 CHECK_EQ(bottom[0]->height(), bottom[0]->width())
      << "Input must be a square matrix for each channel";
}

template <typename Dtype>
void DiagLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int rows = bottom[0]->height();
  top[0]->Reshape(bottom[0]->num(),bottom[0]->channels(),1,rows);
}

template <typename Dtype>
void DiagLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  Dtype* top_data = top[0]->mutable_cpu_data();
  for (int i = 0; i < bottom[0]->num(); i++)
  {
	  
	  for (int j = 0; j < bottom[0]->channels(); j++)
	  {
		  for (int r = 0; r < bottom[0]->height(); r++)
		  {
			  int offset_bottom = bottom[0]->offset(i,j,r,r);
	          int offset_top = top[0]->offset(i,j,1,r);
			  top_data[offset_top] = bottom[0]->cpu_data()[offset_bottom];
			  
		  }
	  }
  }
}

template <typename Dtype>
void DiagLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  caffe_set(bottom[0]->count(), Dtype(0.), bottom_diff);
  for (int i = 0; i < bottom[0]->num(); i++)
  {
	  
	  for (int j = 0; j < bottom[0]->channels(); j++)
	  {
		  for (int r = 0; r < bottom[0]->height(); r++)
		  {
			  int offset_bottom = bottom[0]->offset(i,j,r,r);
	          int offset_top = top[0]->offset(i,j,1,r);
			  bottom_diff[offset_bottom] = top_diff[offset_top];
			  
		  }
	  }
  }
}

#ifdef CPU_ONLY
STUB_GPU(DiagLayer);
#endif

INSTANTIATE_CLASS(DiagLayer);
REGISTER_LAYER_CLASS(Diag);

}  // namespace caffe
