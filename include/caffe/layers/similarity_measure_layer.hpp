#ifndef CAFFE_SIMILARITY_MEASURE_LAYER_HPP_
#define CAFFE_SIMILARITY_MEASURE_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief A "similarity-measure" layer, computes an inner product
 *        with a set of class centers.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class SimilarityMeasureLayer : public Layer<Dtype> {
 public:
  explicit SimilarityMeasureLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "SimilarityMeasure"; }
  virtual inline int ExactNumBottomBlobs() const { return 3; }
  virtual inline int MinTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  //virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    //  const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  //virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
    //  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int M_;
  int K_;
  int N_;
  Dtype loss_weight_;
  unsigned int  video_num_;//input clips num, must identical for all batches
  Blob<Dtype> distance_;
  Blob<Dtype> variation_sum_;
  Blob<Dtype> mean_;// mean of each set 
  Blob<Dtype> dis_; // dis between mean and center
};

}  // namespace caffe

#endif  // CAFFE_SIMILARITY_MEASURE_LAYER_HPP_
