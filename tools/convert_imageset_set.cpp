// This program converts a set of image sets to a lmdb/leveldb by storing them
// as Datum proto buffers.
// Usage:
//   convert_imageset_set [FLAGS] ROOTFOLDER/ LISTFILE SAMPLELISTPARENT/ DB_NAME
//
// where ROOTFOLDER is the root folder that holds all the image sets, each set
// has a folder that holds images belong to that set and LISTFILE
// should be a list of folders of sets, in the format as
//   set_label_origOrder_numOrder, e.g set_0000_00002_00029
//   ....
// speed: about 1min per 1000 sets with size 10 100*100 resized images
#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"

using namespace caffe;  // NOLINT(build/namespaces)
using std::pair;
using boost::scoped_ptr;

//定义一些flags及他们的默认值，对输入做些预处理，[Flags]不计入输入参数数量
DEFINE_bool(gray, false,
    "When this option is on, treat images as grayscale ones");
DEFINE_bool(shuffle, false,
    "Randomly shuffle the order of images and their labels");
DEFINE_string(backend, "lmdb",
        "The backend {lmdb, leveldb} for storing the result");
DEFINE_int32(resize_width, 0, "Width images are resized to");//resize 输入图像
DEFINE_int32(resize_height, 0, "Height images are resized to");
DEFINE_bool(check_size, false,
    "When this option is on, check that all the datum have the same size");
//DEFINE_bool(encoded, false,
  //  "When this option is on, the encoded image will be save in datum");
//DEFINE_string(encode_type, "",
  //  "Optional: What type should we encode the image as ('png','jpg',...).");

int main(int argc, char** argv) {
#ifdef USE_OPENCV  //使用opencv载入图像，预处理等
  ::google::InitGoogleLogging(argv[0]);
  // Print output to stderr (while still logging)
  FLAGS_alsologtostderr = 1;

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("Convert a set of image set to the leveldb/lmdb\n"
        "format used as input for Caffe.\n"
        "Usage:\n"
        "    convert_imageset_set [FLAGS] ROOTFOLDER/ LISTFILE SAMPLELISTPARENT/ DB_NAME\n"
        "The ImageNet dataset for the training demo is at\n"
        "    http://www.image-net.org/download-images\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);//使用gflags对输入命令进行解析，这样可以不把[flags]计入参数数量

  if (argc < 5) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/convert_imageset_set");
    return 1;
  }

  const bool is_color = !FLAGS_gray;
  const bool check_size = FLAGS_check_size;
//  const bool encoded = FLAGS_encoded;
//  const string encode_type = FLAGS_encode_type;

  std::ifstream infile(argv[2]);//读取图像列表文件
  std::vector<std::pair<std::string, int> > lines;//构建<文件名, label >的pair
  std::string filename;
  
  while (infile >> filename) {
    size_t lidx = filename.find('_');
    size_t ridx = filename.find('_',lidx+1);
	size_t ridx3 = filename.rfind('_');
	// 
    int lab_str = std::atoi(filename.substr(lidx+1, ridx-lidx-1).c_str());//sub lab
    int lab_str2 = std::atoi(filename.substr(ridx3+1, filename.length()-ridx3-1).c_str());//seq lab 
	// LOG(INFO)<<filename<<' '<<filename.substr(lidx+1, ridx-lidx-1)<<' '<<filename.substr(ridx3+1, filename.length()-ridx3-1);
	int lab_syns = lab_str2*1000 + lab_str;// lab_syns%1000=lab_str lab_syns/1000=lab_str2
    lines.push_back(std::make_pair(filename, lab_syns));
  }
  infile.close();
  if (FLAGS_shuffle) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    shuffle(lines.begin(), lines.end());//打乱pair的顺序
  }
  LOG(INFO) << "A total of " << lines.size() << " sequences.";

//  if (encode_type.size() && !encoded)
  //  LOG(INFO) << "encode_type specified, assuming encoded=true.";

  int resize_height = std::max<int>(0, FLAGS_resize_height);
  int resize_width = std::max<int>(0, FLAGS_resize_width);

  // Create new DB，根据指定的DB类型创建数据集的DB文件
  scoped_ptr<db::DB> db(db::GetDB(FLAGS_backend));
  db->Open(argv[4], db::NEW);
  scoped_ptr<db::Transaction> txn(db->NewTransaction());//txn为一次交易单

  // Storing to db
  std::string root_folder(argv[1]);
  Datum datum;
  int count = 0;
  int data_size = 0;
  bool data_size_initialized = false;
  std::string parentList(argv[3]);

  for (int line_id = 0; line_id < lines.size(); ++line_id) {
    bool status;//记录对当前的图像转换是否成功
	string listfilename = parentList + lines[line_id].first + "/fileList.txt";// image list file of a seq 
	std::ifstream inSeq(listfilename.c_str());
	string imageName;
	while(inSeq >> imageName){
		size_t index = imageName.find('\\',0);
		imageName = imageName.substr(0,index) + "/" + imageName.substr(index+1,imageName.length()-1);
		//LOG(INFO) << index;
		status = ReadImageToDatum(root_folder + imageName,
        lines[line_id].second, resize_height, resize_width, is_color,
        "", &datum);//核心语句，通过opencv读取指定路径的图像，连同其label，一起存入datum，参考io.cpp文件
        //LOG(INFO)<<datum.label(0)<<" "<<datum.label(1);
        if (status == false) continue;
        if (check_size) {
          if (!data_size_initialized) {
            data_size = datum.channels() * datum.height() * datum.width();
            data_size_initialized = true;
          } 
		  else {
            const std::string& data = datum.data();
            CHECK_EQ(data.size(), data_size) << "Incorrect data field size "
              << data.size();
          }
        }
        // sequential，为当前待入库的datum建立索引：key_str（关键字）
		// the datum is sorted by the key in the DB, very important
        string key_str = caffe::format_int(count, 8) + "_" + lines[line_id].first;

        // Put in db，以字符串形式放入当前交易单中
        string out;
        CHECK(datum.SerializeToString(&out));
        txn->Put(key_str, out);//

        if (++count % 1000 == 0) {//当交易单满1000，提交入库并开启新的交易单
        // Commit db
         txn->Commit();
         txn.reset(db->NewTransaction());
         LOG(INFO) << "Processed " << count << " files.";
        }
	}
	inSeq.close();
  }
  // write the last batch
  if (count % 1000 != 0) {//把最后一单不足1000的交易也提交了
    txn->Commit();
    LOG(INFO) << "Processed " << count << " files.";
  }
#else
  LOG(FATAL) << "This tool requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
  return 0;
}
