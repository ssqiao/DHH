#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/similarity_measure_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SimilarityMeasureLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int num_output = this->layer_param_.similarity_measure_param().num_output();
  N_ = num_output;// class #
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.similarity_measure_param().axis());
  loss_weight_ = this->layer_param_.similarity_measure_param().center_loss_weight();
  // Dimensions starting from "axis" are "flattened" into a single
  // length K_ vector. For example, if bottom[0]'s shape is (N, C, H, W),
  // and axis == 1, N inner products with dimension CHW are performed.
  K_ = bottom[0]->count(axis);
  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    
    this->blobs_.resize(1);
    
    // Initialize the weights
    vector<int> center_shape(2);
    center_shape[0] = N_;
    center_shape[1] = K_;
    this->blobs_[0].reset(new Blob<Dtype>(center_shape));//2*N_*K
    // fill the weights
    shared_ptr<Filler<Dtype> > center_filler(GetFiller<Dtype>(
        this->layer_param_.similarity_measure_param().center_filler()));
    center_filler->Fill(this->blobs_[0].get());
  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void SimilarityMeasureLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[1]->channels(), 2);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
  CHECK_EQ(bottom[2]->channels(), 1);
  CHECK_EQ(bottom[2]->height(), 1);
  CHECK_EQ(bottom[2]->width(), 1);
  CHECK_EQ(bottom[1]->num(), bottom[2]->num());
  video_num_ = bottom[1]->num();
  M_ = bottom[0]->num();
  top[0]->Reshape(M_,1,1,1);
  distance_.ReshapeLike(*bottom[0]);
  variation_sum_.ReshapeLike(*this->blobs_[0]);
  mean_.Reshape(1,1,1,K_);
  dis_.Reshape(1,1,1,K_);
  if(top.size()>1){
	  top[1]->Reshape(1,1,1,1);
  }
}

template <typename Dtype>
void SimilarityMeasureLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* seidx_data = bottom[1]->cpu_data();
  const Dtype* label_data = bottom[2]->cpu_data();
  const Dtype* center_data = this->blobs_[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  Dtype* distance_data = distance_.mutable_cpu_data();
  caffe_set(M_, Dtype(0.), top_data);
  
  
  if (this->phase_ == TRAIN){
	 int count = 0;
     for(int clip_id = 0; clip_id < video_num_; clip_id++)
    {
        int start = seidx_data[clip_id*2];
        int end = seidx_data[clip_id*2+1];
		int label_value = static_cast<int>(label_data[clip_id]);
		count += (end - start + 1);
        for (int frame_id = start; frame_id <= end; frame_id++)
        {
			caffe_sub(K_, bottom_data + frame_id * K_, center_data + label_value * K_, distance_data + frame_id * K_);
			Dtype dot = caffe_cpu_dot(K_, bottom_data + frame_id * K_, center_data + label_value * K_);
			top_data[frame_id] = dot;
        }		
    }
	if(top.size()>1){
		Dtype dot2 = caffe_cpu_dot(count * K_, distance_.cpu_data(), distance_.cpu_data());
	    Dtype loss = dot2 / count / Dtype(2.);
	    top[1]->mutable_cpu_data()[0] = loss_weight_ * loss;
	}
	
  }
  // test phase
  else{
      int count = 0;  	  
	  for(int clip_id = 0; clip_id < video_num_; clip_id++)
     {
        int start = seidx_data[clip_id*2];
        int end = seidx_data[clip_id*2+1];
		Dtype coeff  = 1.0/(end - start + 1);
        count += (end - start + 1);
		// compute the mean of set
		caffe_set(K_,Dtype(0.),mean_.mutable_cpu_data());
        for (int frame_id = start; frame_id <= end; frame_id++)
        {
		   caffe_axpy(K_, Dtype(1.0), bottom_data + frame_id*K_, mean_.mutable_cpu_data());
        }
		caffe_scal(K_, coeff, mean_.mutable_cpu_data());
		// look for the nearest center
		caffe_sub(K_,mean_.cpu_data(), center_data, dis_.mutable_cpu_data());
		Dtype min_dis = caffe_cpu_dot(K_,dis_.cpu_data(), dis_.cpu_data());
		int label_value = 0;
		for (int i = 1; i < N_; i++){
			caffe_sub(K_, mean_.cpu_data(), center_data + i * K_, dis_.mutable_cpu_data());
			Dtype cur_dis = caffe_cpu_dot(K_,dis_.cpu_data(), dis_.cpu_data());
			if (cur_dis < min_dis){
				min_dis = cur_dis;
				label_value = i;
			}
		}
		// compute the similarity
		for (int frame_id = start; frame_id <= end; frame_id++)
        {
			caffe_sub(K_, bottom_data + frame_id * K_, center_data + label_value * K_, distance_data + frame_id * K_);
			Dtype dot = caffe_cpu_dot(K_, bottom_data + frame_id * K_, center_data + label_value * K_);
			top_data[frame_id] = dot;
        }
		if(top.size()>1){
			Dtype dot2 = caffe_cpu_dot(count * K_, distance_.cpu_data(), distance_.cpu_data());
	        Dtype loss = dot2 / count / Dtype(2.);
	        top[1]->mutable_cpu_data()[0] = loss_weight_ * loss;
		}
       		
     }
	  
  }
  
}

template <typename Dtype>
void SimilarityMeasureLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
		
  const Dtype* label_data = bottom[2]->cpu_data();
  const Dtype* seidx_data = bottom[1]->cpu_data();
  const Dtype* distance_data = distance_.cpu_data();
  // Gradient w.r.t class centers  		
  if (this->param_propagate_down_[0]) {
    if (this->phase_ == TRAIN){
        Dtype* center_diff = this->blobs_[0]->mutable_cpu_diff();
        Dtype* variation_sum_data = variation_sum_.mutable_cpu_data();        

        // \sum_{y_i==j}
        caffe_set(N_ * K_, (Dtype)0., variation_sum_.mutable_cpu_data());
        for (int n = 0; n < N_; n++) {
          int count = 0;
		  
          for (int clip_id = 0; clip_id < video_num_; clip_id++) {
		    int start = seidx_data[clip_id*2];
            int end = seidx_data[clip_id*2+1];	  
            const int label_value = static_cast<int>(label_data[clip_id]);
            if (label_value == n) {
              count += (end - start+1);
			  for (int frame_id = start; frame_id <=end; frame_id++){
				caffe_sub(K_, variation_sum_data + n * K_, distance_data + frame_id * K_, variation_sum_data + n * K_);
			  }
            
            }
          }
          caffe_axpy(K_, (Dtype)1./(count + (Dtype)1.), variation_sum_data + n * K_, center_diff + n * K_);
        }
	}
  }
 
  // back propagate to bottom 
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
	const Dtype* center_data = this->blobs_[0]->cpu_data();
	Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
	caffe_set(bottom[0]->count(), Dtype(0.), bottom_diff);
    // Gradient with respect to bottom data
    if (this->phase_ == TRAIN) {
		int count = 0;
      for(int clip_id = 0; clip_id < video_num_; clip_id++)
     {
        int start = seidx_data[clip_id*2];
        int end = seidx_data[clip_id*2+1];
		int label_value = static_cast<int>(label_data[clip_id]);
		count += (end - start + 1);
        for (int frame_id = start; frame_id <= end; frame_id++)
        {
			caffe_cpu_scale(K_, top_diff[frame_id],  center_data + label_value*K_, bottom_diff + frame_id*K_);
        }		
     }
	 if (top.size()>1){
		 caffe_axpy(count * K_, Dtype(loss_weight_ / count), distance_data, bottom_diff);
	 }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(SimilarityMeasureLayer);
#endif

INSTANTIATE_CLASS(SimilarityMeasureLayer);
REGISTER_LAYER_CLASS(SimilarityMeasure);

}  // namespace caffe
