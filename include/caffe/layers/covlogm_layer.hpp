#ifndef CAFFE_COVLOGM_LAYER_HPP_
#define CAFFE_COVLOGM_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Computes the logm for a n*n covariance matrix.
 */
template <typename Dtype>
class CovlogmLayer : public Layer<Dtype> {
 public:
  /**
   * @param param provides LogmParameter logm_param,
   *     with LogmLayer options:
   *   - nothing.
   */
  explicit CovlogmLayer(const LayerParameter& param)
      : Layer<Dtype>(param){}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Covlogm"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  /**
   * @param
   *  bottom: feature matrice of clip F
   *  top: logm of covariance matrice (logm(F'F+epsilon*I))
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
  Blob<Dtype> D_;
  Blob<Dtype> tmp_;
  Blob<Dtype> tmp2_;
  Blob<Dtype> tmp3_;
  Blob<Dtype> logC_;
  Blob<Dtype> K_;
  Blob<Dtype> Unit_;
  Blob<Dtype> unit_;
  Dtype epsilon_;
  Dtype eps_;
  Dtype iter_;
};

}  // namespace caffe

#endif  // CAFFE_COVLOGM_LAYER_HPP_
