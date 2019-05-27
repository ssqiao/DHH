#include <cfloat>
#include <vector>
#include <fstream>
#include "caffe/layers/adapt_temporal_pooling_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void AdaptTemporalPoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  video_num_ = bottom[2]->num();
}

template <typename Dtype>
void AdaptTemporalPoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  top[0]->Reshape(video_num_,bottom[0]->channels(),bottom[0]->height(),bottom[0]->width());//qss
}

template <typename Dtype>
void AdaptTemporalPoolingLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  
  // dim of frame-level features
  const int count = bottom[0]->channels()*bottom[0]->height()*bottom[0]->width();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* weight_data = bottom[1]->cpu_data();
  const Dtype* seidx_data = bottom[2]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  caffe_set(count*video_num_, Dtype(0), top_data);
    // avg pooling for each clip
    for(int clip_id = 0; clip_id < video_num_; clip_id++)
    {
        int start = seidx_data[clip_id*2];
        int end = seidx_data[clip_id*2+1];
        int offset_top = top[0]->offset(clip_id);
        for (int frame_id = start; frame_id <= end; frame_id++)
        {
			Dtype coeff = weight_data[frame_id];
            int offset_bottom = bottom[0]->offset(frame_id);
            caffe_axpy(count, coeff, bottom_data+offset_bottom, top_data+offset_top);
        }		
    }

}

template <typename Dtype>
void AdaptTemporalPoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  
      const int count = bottom[0]->channels()*bottom[0]->height()*bottom[0]->width();
      const Dtype* top_diff = top[0]->cpu_diff();
      const Dtype* seidx_data = bottom[2]->cpu_data();
      Dtype* bottom_diff_data = bottom[0]->mutable_cpu_diff();
	  Dtype* bottom_diff_weight = bottom[1]->mutable_cpu_diff();
      caffe_set(bottom[0]->count(), Dtype(0), bottom_diff_data);
      caffe_set(bottom[1]->count(), Dtype(0), bottom_diff_weight);
      for(int clip_id = 0; clip_id < video_num_; clip_id++)
     {
        int start = seidx_data[clip_id*2];
        int end = seidx_data[clip_id*2+1];
        int offset_top = top[0]->offset(clip_id);
          for (int frame_id = start; frame_id <= end; frame_id++)
         {
			Dtype coeff = bottom[1]->cpu_data()[frame_id]; 
            int offset_bottom = bottom[0]->offset(frame_id);
            caffe_cpu_scale(count, coeff, top_diff+offset_top, bottom_diff_data+offset_bottom);
			bottom_diff_weight[frame_id] = caffe_cpu_dot(count, top_diff+offset_top, bottom[0]->cpu_data()+offset_bottom);
         }
			  
     }      

   
}

#ifdef CPU_ONLY
STUB_GPU(AdaptTemporalPoolingLayer);
#endif

INSTANTIATE_CLASS(AdaptTemporalPoolingLayer);
REGISTER_LAYER_CLASS(AdaptTemporalPooling);

}  // namespace caffe
