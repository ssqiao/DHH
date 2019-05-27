#ifndef CAFFE_ADAPT_TEMPORAL_POOLING_LAYER_HPP_
#define CAFFE_ADAPT_TEMPORAL_POOLING_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Compute temporal pooling operations, such as average and max pooling,
 *        within each input clip.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class AdaptTemporalPoolingLayer : public Layer<Dtype> {
 public:
  explicit AdaptTemporalPoolingLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "AdaptTemporalPooling"; }
  virtual inline int ExactNumBottomBlobs() const { return 3; }// data , weight ,indicator label
  virtual inline int ExactNumTopBlobs() const { return 1; }//  pooled data

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
    // not implemented yet
  //virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
     // const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
    // not implemented yet
  //virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
     // const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  unsigned int  video_num_;//input clips num, must identical for all batches

};

}  // namespace caffe

#endif  // CAFFE_ELTWISE_LAYER_HPP_
