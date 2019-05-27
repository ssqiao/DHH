#ifndef CAFFE_TEMPORAL_POOLING_LAYER_HPP_
#define CAFFE_TEMPORAL_POOLING_LAYER_HPP_

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
class TemporalPoolingLayer : public Layer<Dtype> {
 public:
  explicit TemporalPoolingLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "TemporalPooling"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }// data and indicator label
  virtual inline int MaxTopBlobs() const { return 4; }// one for pooled data, one for merged labels

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

  TemporalPoolingParameter_TemPoolOp op_;
  unsigned int  video_num_;//input clips num, must identical for all batches
  Blob<Dtype> start_end_idx_;//start and end index for each clip in the batch
  Blob<Dtype> sampling_idx_; //idx of sampled idx
  //when the op_ is MAX pooling, record the max index of each dimension for each clip, so the blob's size is (video_num_, channels, height, width
  Blob<int>  max_idx_;

};

}  // namespace caffe

#endif  // CAFFE_ELTWISE_LAYER_HPP_
