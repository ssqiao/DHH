#ifndef CAFFE_LOGM_LAYER_HPP_
#define CAFFE_LOGM_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Computes the logm for a n*n covariance matrix.
 */
template <typename Dtype>
class LogmLayer : public Layer<Dtype> {
 public:
  /**
   * @param param provides LogmParameter logm_param,
   *     with LogmLayer options:
   *   - nothing.
   */
  explicit LogmLayer(const LayerParameter& param)
      : Layer<Dtype>(param){}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Logm"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  /**
   * @param
   *  bottom: covariance matrices
   *  top: logm of covariance matrices
   */
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  //virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    //  const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  //virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
    //  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  /// store the res of SVD decomposition and logm.
  Blob<Dtype> S_;
  Blob<Dtype> singular_;
  Blob<Dtype> U_;
  Blob<Dtype> Vt_;
  //Blob<Dtype> Res_;
  Blob<Dtype> tmp_;
  Blob<Dtype> tmp2_;
  Blob<Dtype> tmp3_;
  Blob<Dtype> logS_;
  Blob<Dtype> P_;
  Blob<Dtype> Unit_;
};

}  // namespace caffe

#endif  // CAFFE_LOGM_LAYER_HPP_
