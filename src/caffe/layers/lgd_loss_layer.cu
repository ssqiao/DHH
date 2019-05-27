#include <vector>

#include "caffe/layers/lgd_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void LgdLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, 
    const vector<Blob<Dtype>*>& top) {
  const int count = bottom[0]->count()/bottom[0]->num();
  const Dtype alpha = 4*top[0]->cpu_diff()[0] / bottom[0]->num()/(bottom[0]->num()-1)/(bottom[0]->num()-2)/(bottom[0]->num()+1);
  caffe_gpu_set(bottom[0]->count(),Dtype(0.),bottom[0]->mutable_gpu_diff());
  //caffe_gpu_set(diff_.count(),Dtype(0.),diff_.mutable_gpu_data());
  //caffe_gpu_set(realLoss_.count(),Dtype(0.),realLoss_.mutable_gpu_data());
  Dtype loss = 0.;
  const Dtype epsilon = 0.0001;
  const Dtype lamda = 1.0;
  for(int i = 0; i < bottom[0]->num(); i++) {
	  int offset_a = bottom[0]->offset(i);
	  for(int j = i+1; j < bottom[0]->num(); j++) {
		  int offset_b = bottom[0]->offset(j);
		  caffe_gpu_sub(
          count,
          bottom[0]->gpu_data()+offset_a,
          bottom[0]->gpu_data()+offset_b,
          diff1_.mutable_gpu_data());
		  
          Dtype dot1 = 0.;
		  caffe_gpu_dot(count, diff1_.gpu_data(), diff1_.gpu_data(),&dot1);
		  Dtype tmp1 = bottom[1]->cpu_data()[i*bottom[0]->num()+j]+epsilon;
		  dot1 = (dot1+lamda)/tmp1;
		  for (int m = i; m < bottom[0]->num();m++) {
			  int offset_c = bottom[0]->offset(m);
			  for (int n = m+1; n < bottom[0]->num();n++) {
				  if (m*bottom[0]->num()+n <= i*bottom[0]->num()+j)
					  continue;
				  int offset_d = bottom[0]->offset(n);
				  caffe_gpu_sub(
                  count,
                  bottom[0]->gpu_data()+offset_c,
                  bottom[0]->gpu_data()+offset_d,
                  diff2_.mutable_gpu_data());
                  Dtype dot2 = 0.;
				  Dtype tmp2 = bottom[1]->cpu_data()[m*bottom[0]->num()+n]+epsilon;
				  caffe_gpu_dot(count, diff2_.gpu_data(), diff2_.gpu_data(),&dot2);
		          dot2 = (dot2+lamda)/tmp2;
				  
				  Dtype diff_ij_mn = dot1 - dot2;
                  loss += abs(diff_ij_mn);	

                  Dtype sign = (diff_ij_mn >= 0) ? 1 : -1;
                  caffe_gpu_axpby(
			      count,
			      2*alpha*sign/tmp1,
			      diff1_.gpu_data(),
			      Dtype(1.),
			      bottom[0]->mutable_gpu_diff()+offset_a);				  
				  
				  caffe_gpu_axpby(
			      count,
			      -2*alpha*sign/tmp1,
			      diff1_.gpu_data(),
			      Dtype(1.),
			      bottom[0]->mutable_gpu_diff()+offset_b);
				  
				  caffe_gpu_axpby(
			      count,
			      -2*alpha*sign/tmp2,
			      diff2_.gpu_data(),
			      Dtype(1.),
			      bottom[0]->mutable_gpu_diff()+offset_c);
				  
				  caffe_gpu_axpby(
			      count,
			      2*alpha*sign/tmp2,
			      diff2_.gpu_data(),
			      Dtype(1.),
			      bottom[0]->mutable_gpu_diff()+offset_d);
			  }
		  }	
	  }
  }
  
  loss = loss * alpha;
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void LgdLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }	
}

INSTANTIATE_LAYER_GPU_FUNCS(LgdLossLayer);

}  // namespace caffe
