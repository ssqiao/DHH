#include <cfloat>
#include <vector>
#include <algorithm>
#include <fstream>
#include "caffe/layers/covariance_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
 
template <typename Dtype>
__global__ void NonSingularize(const int nthreads, Dtype* top_data, const Dtype miu) {
  Dtype trace = 0;
  for(int i = 0; i < nthreads; i++){
     trace = trace + top_data[i*nthreads + i];
  } 
  CUDA_KERNEL_LOOP(index, nthreads) {
	top_data[index*nthreads + index] = top_data[index*nthreads + index] + miu*trace;
  }
}
template <typename Dtype>
__global__ void OuterProductBackward(const int nthreads, const Dtype* diff,
     Dtype* kron) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    Dtype val = diff[index];
	int offset_kron = index*nthreads*nthreads;
	
	for(int index1 = 0; index1 < nthreads; index1++) {
	  kron[offset_kron+index1*nthreads+index1] = val;// diagnal elements 
	  kron[offset_kron+index1*nthreads+index] = kron[offset_kron+index1*nthreads+index] + diff[index1];
	}
  }
}
template <typename Dtype>
__global__ void nonsingularizeBackward(const int nthreads, const Dtype* diff,
     Dtype* kron, const Dtype miu) {
  CUDA_KERNEL_LOOP(index, nthreads) {
	int offset_kron = index*(nthreads+1)*nthreads;
	
	for(int index1 = 0; index1 < nthreads; index1++) {
	  kron[offset_kron+index1] = kron[offset_kron+index1] + 2*miu*diff[index1];
	}
  }
}

template <typename Dtype>
void CovarianceLayer<Dtype>::Forward_gpu(
        const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    Dtype  first_label = bottom[1]->cpu_data()[1];
    Dtype* seidx_data = start_end_idx_.mutable_cpu_data();
    Dtype* mean_data = mean_.mutable_gpu_data();
    Dtype* label_data = NULL;
    if(top.size()>1){
        label_data = top[1]->mutable_cpu_data();
        label_data[0] = bottom[1]->cpu_data()[0];
    }
    
    // dim of frame feature
    const int count = bottom[0]->channels()*bottom[0]->height()*bottom[0]->width();
    caffe_gpu_set(mean_.count(), Dtype(0.), mean_data);
    
	if(bottom.size()>2){
		caffe_copy(bottom[1]->count(),bottom[1]->cpu_data(),label_data);
		caffe_copy(bottom[2]->count(),bottom[2]->cpu_data(),seidx_data);
		for(int clip_idx = 1; clip_idx < video_num_; clip_idx++){
			int start = start_end_idx_.cpu_data()[clip_idx*2];
            int end = start_end_idx_.cpu_data()[clip_idx*2+1];
            Dtype coeff = 1.0 / (end - start);
            int offset_mean = mean_.offset(clip_idx);
            for (int frame_idx = start; frame_idx <= end; frame_idx++)
           {
              int offset_bottom = bottom[0]->offset(frame_idx);
              // fi - m --> differ_
              caffe_gpu_axpy(count, Dtype(1.0), bottom[0]->cpu_data()+offset_bottom, mean_data+offset_mean);
           }
		   caffe_gpu_scal(count, coeff, mean_data+offset_mean);
		}
	}
	else{
	    seidx_data[0] = 0;
        int clip_count = 1;
	
        int offset_mean = mean_.offset(clip_count-1);
        int offset_bottom = bottom[0]->offset(0);
        // sum each clip's frame features
        caffe_gpu_axpy(count, Dtype(1.0), bottom[0]->gpu_data()+offset_bottom, mean_data+offset_mean);
	
        // calaculate m = 1/n * sum (fi) for each clip
        for(int i = 1; i < bottom[1]->num(); i++)
       {
           seidx_data[clip_count*2-1] = i;
           if(bottom[1]->cpu_data()[i*2+1] != first_label)// the end of the previous clip and begin of a new clip
          {
            seidx_data[clip_count*2-1] = i-1;// fix the end index of previous clip
            Dtype coeff  = 1.0/(seidx_data[clip_count*2-1] -seidx_data[clip_count*2-2] + 1);
            // average frame features of previous clip
            caffe_gpu_scal(count, coeff, mean_data+offset_mean);
            if (clip_count < video_num_)// config the new clip
            {
                seidx_data[clip_count*2] = i;
                first_label = bottom[1]->cpu_data()[i*2+1];
                if(top.size()>1) {
                    label_data[clip_count] = bottom[1]->cpu_data()[i*2];
                }
                clip_count ++;
                offset_mean = mean_.offset(clip_count-1);
            }
            else// all clips have been found
            {
                break;
            }
          }
          offset_bottom = bottom[0]->offset(i);
          caffe_gpu_axpy(count, Dtype(1.0), bottom[0]->gpu_data()+offset_bottom, mean_data+offset_mean);
       }
	
       // all clips just fullfill this batch, no redundant frames
       if(seidx_data[clip_count*2-1] == bottom[1]->num() -1 ){
          Dtype coeff  = 1.0/(seidx_data[clip_count*2-1] -seidx_data[clip_count*2-2] + 1);
          caffe_gpu_scal(count, coeff, mean_data+offset_mean);
       }
	
	
	}
	
	
    
	

    // calculate C = 1/(n-1) * sum {(fi-m)*(fi-m)'} for each clip.
    Dtype* top_data = top[0]->mutable_gpu_data();
    caffe_gpu_set(top[0]->count(), Dtype(0.), top_data);
    // covariance for each clip
    for(int clip_id = 0; clip_id < video_num_; clip_id++)
    {
        int start = start_end_idx_.cpu_data()[clip_id*2];
        int end = start_end_idx_.cpu_data()[clip_id*2+1];
        Dtype coeff = 1.0 / (end - start);
        int offset_top = top[0]->offset(clip_id);
        int offset_mean = mean_.offset(clip_id);
        for (int frame_id = start; frame_id <= end; frame_id++)
        {
            int offset_bottom = bottom[0]->offset(frame_id);
            // fi - m --> differ_
            caffe_gpu_sub(count,bottom[0]->gpu_data()+offset_bottom, mean_data+offset_mean, differ_.mutable_gpu_data() );
            caffe_gpu_gemm<Dtype>(CblasNoTrans,CblasNoTrans, count,count,1, coeff, differ_.gpu_data(),differ_.gpu_data(),Dtype(1),top_data+offset_top);
        }
        // make sure the nonsingular property of the covariance, add a small eye matrix
        //for (int row = 0; row < count; row++){//可以并行加速
        //   caffe_gpu_add_scalar(1,Dtype(miu_),top_data+offset_top+row*count+row);
        //}
		
		NonSingularize<Dtype> <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count,top_data+offset_top,miu_);
		
    }
}


template <typename Dtype>
void CovarianceLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                          const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    
    for (int i = 0; i < bottom.size(); ++i) {
        if (propagate_down[i]) {
		    const int count = bottom[i]->channels()*bottom[i]->height()*bottom[i]->width();
            const Dtype* top_diff = top[i]->gpu_diff();
            Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
            caffe_gpu_set(bottom[i]->count(), Dtype(0.), bottom_diff);
            const Dtype* mean_data = mean_.gpu_data();
                for(int clip_id = 0; clip_id < video_num_; clip_id++)
                {
                    int start = start_end_idx_.cpu_data()[clip_id*2];
                    int end = start_end_idx_.cpu_data()[clip_id*2+1];
                    Dtype coeff = 1.0 / (end - start);
                    int offset_top = top[i]->offset(clip_id);
                    int offset_mean = mean_.offset(clip_id);
                    for (int frame_id = start; frame_id <= end; frame_id++) 
                    {
                        int offset_bottom = bottom[i]->offset(frame_id);
                        caffe_gpu_sub(count,bottom[i]->gpu_data()+offset_bottom, mean_data+offset_mean, differ_.mutable_gpu_data());

                        // calc the kronecker product kron(differ_',I) + kron(I, differ_')
                        caffe_gpu_set(kron_.count(),Dtype(0.),kron_.mutable_gpu_data());
						
						// need for speed
						OuterProductBackward<Dtype> <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count,differ_.gpu_data(),kron_.mutable_gpu_data());
					    nonsingularizeBackward<Dtype> <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count,differ_.gpu_data(),kron_.mutable_gpu_data(),miu_);

                        caffe_gpu_gemm<Dtype>(CblasNoTrans,CblasNoTrans,1,count,count*count,coeff,top_diff+offset_top,kron_.gpu_data(),Dtype(0.),bottom_diff+offset_bottom) ;
                    }
                }

        }
    }
}

INSTANTIATE_LAYER_GPU_FUNCS(CovarianceLayer);

}  // namespace caffe
