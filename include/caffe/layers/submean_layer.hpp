#ifndef CAFFE_SUBMEAN_LAYER_HPP_
#define CAFFE_SUBMEAN_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Compute covariance operations, i.e., 1/(n-1)*sum(fi-f)*(fi-f)',
 *        along each input clip.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class SubmeanLayer : public Layer<Dtype> {
 public:
  explicit SubmeanLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Submean"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }// data and indicator label
  virtual inline int MinTopBlobs() const { return 2; }// normalized data, index for each clip 
  virtual inline int MaxTopBlobs() const { return 5; }// normalized data, index for each clip and label_v, still image, label_s
 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  //virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    //  const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  //virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
    //  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  unsigned int  video_num_;//input clips num, must identical for all batches
  Blob<Dtype> start_end_idx_;//start and end index for each clip in the batch
  // store the mean vectors for each clips, be used when backward
  Blob<Dtype>  mean_;
  unsigned int *index;
  //Blob<Dtype>  differ_;
  //Blob<Dtype>  eye_;// unit matrix
};

}  // namespace caffe

#endif  // CAFFE_SUBMEAN_LAYER_HPP_
