#include <functional>
#include <utility>
#include <vector>
#include <fstream>

#include "caffe/layers/logm_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void LogmLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom.size(), top.size())
      << "Inputs and output must have the same # of blobs.";
}

template <typename Dtype>
void LogmLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const int num = bottom[0]->num();
  const int row = sqrt(bottom[0]->count()/bottom[0]->num());
  top[0]->ReshapeLike(*bottom[0]);
  singular_.Reshape(1,1,1,row);
  U_.Reshape(num,1,row,row);
  Vt_.Reshape(num,1,row,row);
  tmp_.Reshape(1,1,row,row);
  tmp2_.Reshape(1,1,row,row);
  tmp3_.Reshape(1,1,row,row);
  S_.Reshape(num,1,row,row);
  logS_.Reshape(num,1,row,row);
  P_.Reshape(1,1,row,row);
  Unit_.Reshape(1,1,row,row);
  caffe_set(tmp_.count(),Dtype(1.0),tmp_.mutable_cpu_data());
  caffe_cpu_diag_op(row,row,tmp_.cpu_data(),Unit_.mutable_cpu_data(),0);
  //Res_.ReshapeLike(*bottom[0]);
 //fstream output;
 //output.open("/home/data/qiaoshishi/tmp.txt",ios::out);
 //for(int i = 0; i < row; i++)
 //{
 // for(int j = 0; j < row; j++)
 // {
 //	  output<<Unit_.cpu_data()[i*row+j]<<'\t';
 // }
 // output<<std::endl;
 //}
 //output.close();
}

template <typename Dtype>
void LogmLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const int num = bottom[0]->num();
  const int row = sqrt(bottom[0]->count()/bottom[0]->num());
  
  Dtype* top_data = top[0]->mutable_cpu_data();
  Dtype* logS_data = logS_.mutable_cpu_data();
  Dtype* S_data = S_.mutable_cpu_data();
  Dtype* sigular_data = singular_.mutable_cpu_data();
  Dtype* U_data = U_.mutable_cpu_data();
  Dtype* Vt_data = Vt_.mutable_cpu_data();
  Dtype* tmp_data = tmp_.mutable_cpu_data();
  // svd may change the input data
  caffe_copy(bottom[0]->count(),bottom_data,top_data);
  
  caffe_set(S_.count(),Dtype(0.),S_data);
  // compute the logm of each covariance
  for (int i = 0; i < num; i++){
	  int offset = bottom[0]->offset(i);
	  caffe_cpu_gesvd(row,row,top_data+offset,sigular_data,U_data+offset,Vt_data+offset);
	  caffe_cpu_diag(row, row, row, sigular_data, S_data+offset);
	  caffe_cpu_diag_op(row,row,S_data+offset,logS_data+offset,2);
	  caffe_cpu_gemm<Dtype>(CblasNoTrans,CblasNoTrans,row,row,row,Dtype(1.),U_data+offset,logS_data+offset,Dtype(0.),tmp_data);
	  caffe_cpu_gemm<Dtype>(CblasNoTrans,CblasNoTrans,row,row,row,Dtype(1.),tmp_data,Vt_data+offset,Dtype(0.),top_data+offset);
  }
  /* fstream output1, output2, output4, output6;
  output1.open("/home/data/qiaoshishi/data_in.txt",ios::out);
  output2.open("/home/data/qiaoshishi/data_out.txt",ios::out);
  //output3.open("/home/data/qiaoshishi/data_u.txt",ios::out);
  output4.open("/home/data/qiaoshishi/data_s.txt",ios::out);
  //output5.open("/home/data/qiaoshishi/data_vt.txt",ios::out);
  output6.open("/home/data/qiaoshishi/data_logs.txt",ios::out);
  for(int id = 0; id < num; id++)
  {
	  int offset_tmp = bottom[0]->offset(id);
	  for(int m =0; m < row; m++){
		  for(int n =0; n < row; n++){
			  output1<<bottom[0]->cpu_data()[offset_tmp+m*row+n]<<'\t';
			  output2<<top[0]->cpu_data()[offset_tmp+m*row+n]<<'\t';
			  //output3<<U_.cpu_data()[offset_tmp+m*row+n]<<'\t';
			  output4<<S_.cpu_data()[offset_tmp+m*row+n]<<'\t';
			  //output5<<Vt_.cpu_data()[offset_tmp+m*row+n]<<'\t';
			  output6<<logS_.cpu_data()[offset_tmp+m*row+n]<<'\t';
		  }
		  output1<<std::endl;
	      output2<<std::endl;
		  //output3<<std::endl;
		  output4<<std::endl;
		  //output5<<std::endl;
		  output6<<std::endl;
	  }
	  output1<<std::endl;
	  output2<<std::endl;
	  //output3<<std::endl;
      output4<<std::endl;
	  //output5<<std::endl;
	  output6<<std::endl;
  }
  output1.close();
  output2.close();
  //output3.close();
  output4.close();
  //output5.close();
  output6.close(); */
  
}

template <typename Dtype>
void LogmLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                          const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* Unit_data = Unit_.cpu_data();
  const Dtype* U_data = U_.cpu_data();
  const Dtype* logS_data = logS_.cpu_data();
  const Dtype* S_data = S_.cpu_data();  
  const Dtype* Vt_data = Vt_.cpu_data();
  Dtype* P_data = P_.mutable_cpu_data();
  Dtype* tmp_data = tmp_.mutable_cpu_data();
  Dtype* tmp2_data = tmp2_.mutable_cpu_data();
  Dtype* tmp3_data = tmp3_.mutable_cpu_data();
  
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  caffe_set(bottom[0]->count(), Dtype(0.0), bottom_diff);
  
  const int num = bottom[0]->num();
  const int row = sqrt(bottom[0]->count()/bottom[0]->num());
											  
  	for(int clip_id = 0; clip_id < num; clip_id++)
   {
       int offset_top = top[0]->offset(clip_id);          
       // 2 times of sym of the top gradient
       caffe_cpu_gemm<Dtype>(CblasTrans,CblasNoTrans,row,row,row,Dtype(1.),top_diff+offset_top, Unit_data, Dtype(0.),tmp_data);
       caffe_add(row*row, top_diff+offset_top, tmp_data, tmp2_data);

       // gradient of L w.r.t U 
       caffe_cpu_gemm<Dtype>(CblasNoTrans,CblasNoTrans,row,row,row,Dtype(1.),tmp2_data,U_data+offset_top, Dtype(0.),tmp_data);
       caffe_cpu_gemm<Dtype>(CblasNoTrans,CblasNoTrans,row,row,row,Dtype(1.),tmp_data,logS_data+offset_top, Dtype(0.),tmp3_data); 	   
	   
       // create P'
       for(int i = 0; i < row; i++){
		   for(int j = i; j < row; j++){
			   if(i==j)
				   P_data[i*row+j] = 0;
			   else{
				   P_data[i*row+j] = 1.0/(S_data[offset_top+j*row+j] - S_data[offset_top+i*row+i]);
				   P_data[j*row+i] = -P_data[i*row+j];
			   }
		   }
	   }
	   
	   
	   caffe_cpu_gemm<Dtype>(CblasNoTrans,CblasNoTrans,row,row,row,Dtype(1.0),Vt_data+offset_top,tmp3_data,Dtype(0.),tmp_data);
	   
	   // 2 times of sym 
	   caffe_cpu_gemm<Dtype>(CblasTrans,CblasNoTrans,row,row,row,Dtype(1.),tmp_data, Unit_data, Dtype(0.),tmp3_data);
       caffe_add(row*row, tmp3_data, tmp_data, tmp_data);
	
	   // Hadamard product (the bug is here, numerical problem)
	   caffe_mul(row*row,P_data,tmp_data,tmp3_data);
	   
	   // the first term 
	   caffe_cpu_gemm<Dtype>(CblasNoTrans,CblasNoTrans,row,row,row,Dtype(1.0),U_data+offset_top,tmp3_data,Dtype(0.),tmp_data);
	   //caffe_cpu_gemm<Dtype>(CblasNoTrans,CblasNoTrans,row,row,row,Dtype(1.0),tmp_data,P_data,Dtype(0.),tmp3_data);
	   caffe_cpu_gemm<Dtype>(CblasNoTrans,CblasNoTrans,row,row,row,Dtype(1.0),tmp_data,Vt_data+offset_top,Dtype(0.),P_data);
	   
	   // gradient of L w.r.t sigma
	   caffe_cpu_diag_op(row,row,S_data+offset_top, tmp_data,1);
	   caffe_cpu_gemm<Dtype>(CblasNoTrans,CblasNoTrans,row,row,row,Dtype(1.0),tmp_data,Vt_data+offset_top,Dtype(0.),tmp3_data);
	   caffe_cpu_gemm<Dtype>(CblasNoTrans,CblasNoTrans,row,row,row,Dtype(0.5),tmp3_data,tmp2_data,Dtype(0.),tmp_data);
	   caffe_cpu_gemm<Dtype>(CblasNoTrans,CblasNoTrans,row,row,row,Dtype(1.0),tmp_data,U_data+offset_top,Dtype(0.),tmp3_data);
	   
	   
	   //the second term
	   caffe_cpu_diag_op(row,row,tmp3_data,tmp_data,0);
	   caffe_cpu_gemm<Dtype>(CblasNoTrans,CblasNoTrans,row,row,row,Dtype(1.0),U_data+offset_top,tmp_data,Dtype(0.),tmp3_data);
	   caffe_cpu_gemm<Dtype>(CblasNoTrans,CblasNoTrans,row,row,row,Dtype(1.0),tmp3_data,Vt_data+offset_top,Dtype(0.),tmp_data);
	  
	   
	   // sum the first and second term 
	   caffe_add(row*row,P_data,tmp_data,bottom_diff+offset_top);
	   
   }									                 
 
}

#ifdef CPU_ONLY
STUB_GPU(LogmLayer);
#endif

INSTANTIATE_CLASS(LogmLayer);
REGISTER_LAYER_CLASS(Logm);

}  // namespace caffe
