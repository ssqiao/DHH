#include <boost/thread.hpp>
#include <map>
#include <string>
#include <vector>
#include <fstream>

#include "caffe/common.hpp"
#include "caffe/data_reader.hpp"
#include "caffe/layers/data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/rng.hpp"

namespace caffe {

using boost::weak_ptr;

map<const string, weak_ptr<DataReader::Body> > DataReader::bodies_;
static boost::mutex bodies_mutex_;

DataReader::DataReader(const LayerParameter& param)
    : queue_pair_(new QueuePair(  //
                                  param.data_param().prefetch() * param.data_param().batch_size())) {
    // Get or create a body
    boost::mutex::scoped_lock lock(bodies_mutex_);
    string key = source_key(param);
    weak_ptr<Body>& weak = bodies_[key];
    body_ = weak.lock();
    if (!body_) {
        body_.reset(new Body(param));
        bodies_[key] = weak_ptr<Body>(body_);
    }
    body_->new_queue_pairs_.push(queue_pair_);
}

DataReader::~DataReader() {
    string key = source_key(body_->param_);
    body_.reset();
    boost::mutex::scoped_lock lock(bodies_mutex_);
    if (bodies_[key].expired()) {
        bodies_.erase(key);
    }
}

//

DataReader::QueuePair::QueuePair(int size) {
    // Initialize the free queue with requested number of datums
    for (int i = 0; i < size; ++i) {
        free_.push(new Datum());
    }
}

DataReader::QueuePair::~QueuePair() {
    Datum* datum;
    while (free_.try_pop(&datum)) {
        delete datum;
    }
    while (full_.try_pop(&datum)) {
        delete datum;
    }
}

//

DataReader::Body::Body(const LayerParameter& param)
    : param_(param),
      new_queue_pairs_() {
    StartInternalThread();
}

DataReader::Body::~Body() {
    StopInternalThread();
}

void DataReader::Body::InternalThreadEntry() {

    const int class_num_per_batch = param_.data_param().class_per_batch();
    const int clip_num_per_class = param_.data_param().clip_per_class();
    const int batch_size = param_.data_param().batch_size();
    CHECK_GE(class_num_per_batch,1);
    CHECK_GE(clip_num_per_class,1);
    //CHECK_LE(class_num_per_batch*clip_num_per_class*53,batch_size);
	const string still_source = param_.data_param().still_source();
	bool cross_model = false;
	if(!still_source.empty()){
		cross_model = true;
	}
	LOG(INFO)<<"cross:"<<cross_model;
	
    int solver_count = param_.phase() == TRAIN ? Caffe::solver_count() : 1;
    vector<int> frame_count(solver_count,0);
    int clip_count;

    // read the db file one by one in order
    if (class_num_per_batch == 1){
        //open the DB file
        shared_ptr<db::DB> db(db::GetDB(param_.data_param().backend()));
        db->Open(param_.data_param().source(), db::READ);
        shared_ptr<db::Cursor> cursor(db->NewCursor());
        vector<shared_ptr<QueuePair> > qps;
		
		shared_ptr<db::DB> db_still(db::GetDB(param_.data_param().backend()));
		shared_ptr<db::Cursor> cursor_still;
		if(cross_model){
			db_still->Open(param_.data_param().still_source(), db::READ);
			cursor_still = shared_ptr<db::Cursor>(db_still->NewCursor());
		}

        try {
            // To ensure deterministic runs, only start running once all solvers
            // are ready. But solvers need to peek on one item during initialization,
            // so read one item, then wait for the next solver.
            for (int i = 0; i < solver_count; ++i) {
                shared_ptr<QueuePair> qp(new_queue_pairs_.pop());
                read_one(cursor.get(), qp.get(), frame_count[i]);
				if(cross_model){
					read_one(cursor_still.get(), qp.get());
					frame_count[i]++;
				}
                qps.push_back(qp);
            }
            clip_count = 1;

            // Main loop
            while (!must_stop()) {
                // already be enough to fulfill a batch, clear the count value, get ready for next batch
                if(clip_count > clip_num_per_class){
                    clip_count = 0;
                    frame_count.clear();
                    frame_count.assign(solver_count,0);
                }
                bool do_pad = (clip_count==clip_num_per_class) ? true : false;
                for (int i = 0; i < solver_count; ++i) {
                    int tmp_size = (do_pad) ? (batch_size -frame_count[i]) : 0;
                    read_one(cursor.get(), qps[i].get(),tmp_size,do_pad);					
                    frame_count[i] += tmp_size;
					if(cross_model&&!do_pad){
					  read_one(cursor_still.get(), qps[i].get());
					  frame_count[i]++;
				    }
                }
                clip_count ++;
                // Check no additional readers have been created. This can happen if
                // more than one net is trained at a time per process, whether single
                // or multi solver. It might also happen if two data layers have same
                // name and same source.
                CHECK_EQ(new_queue_pairs_.size(), 0);
            }
        } catch (boost::thread_interrupted&) {
            // Interrupted exception is expected on shutdown
        }

    }
    // read the fixed num of clips for the selected classes
    else{
        // get each class's cursor in the DB
        shared_ptr<db::DB> db(db::GetDB(param_.data_param().backend()));
        db->Open(param_.data_param().source(), db::READ);
        shared_ptr<db::Cursor> cursor(db->NewCursor());
        vector<classInfo> class_db_reader_info;

        int first_class_idx = -1;
        classInfo current_class;
        current_class.sample_num = 0;

        while (cursor->valid())
        {
            std::string key_tmp = cursor->key();
            std::string class_label = key_tmp.substr(13, 4);
            int class_idx = atoi(class_label.c_str());

            if (class_idx != first_class_idx)
            {
                if (first_class_idx != -1)
                {
                    class_db_reader_info.push_back(current_class);
                }
                first_class_idx = class_idx;
                current_class.first_key = key_tmp;
                current_class.current_key = key_tmp;
                current_class.sample_num = 0;
                //current_class.sample_count = 0;

            }
            current_class.last_key = key_tmp;
            current_class.sample_num++;
            cursor->Next();
        }
        if (current_class.sample_num != 0)
        {
            class_db_reader_info.push_back(current_class);
        }
        LOG(INFO) << class_db_reader_info.size() << "classes found";
        for (int i = 0; i < class_db_reader_info.size();i++)
        {
            class_db_reader_info[i].cursor = shared_ptr<db::Cursor>(db->NewCursor());
            retrieval_one(class_db_reader_info[i].cursor.get(), class_db_reader_info[i].first_key);
            LOG(INFO) << "class " << i << " has " << class_db_reader_info[i].sample_num << " samples";
        }
        cursor->SeekToFirst();
		
		//for still image DB 
		shared_ptr<db::DB> db_still(db::GetDB(param_.data_param().backend()));
		shared_ptr<db::Cursor> cursor_still;
		vector<classInfo> class_db_reader_info_still;
		if(cross_model){
			db_still->Open(param_.data_param().still_source(), db::READ);
			cursor_still = shared_ptr<db::Cursor>(db_still->NewCursor());
			first_class_idx = -1;
			current_class.sample_num = 0;
			
			while (cursor_still->valid())
           {
              std::string key_tmp = cursor_still->key();
              std::string class_label = key_tmp.substr(13, 4);//need to be adaptive
              int class_idx = atoi(class_label.c_str());

              if (class_idx != first_class_idx)
             {
                if (first_class_idx != -1)
                {
                    class_db_reader_info_still.push_back(current_class);
                }
                first_class_idx = class_idx;
                current_class.first_key = key_tmp;
                current_class.current_key = key_tmp;
                current_class.sample_num = 0;
                //current_class.sample_count = 0;

             }
            current_class.last_key = key_tmp;
            current_class.sample_num++;
            cursor_still->Next();
          }
          if (current_class.sample_num != 0)
         {
            class_db_reader_info_still.push_back(current_class);
         }
          LOG(INFO) << class_db_reader_info_still.size() << "classes found in still db";
          for (int i = 0; i < class_db_reader_info_still.size();i++)
         {
            class_db_reader_info_still[i].cursor = shared_ptr<db::Cursor>(db_still->NewCursor());
            retrieval_one(class_db_reader_info_still[i].cursor.get(), class_db_reader_info_still[i].first_key);
            LOG(INFO) << "class " << i << " has " << class_db_reader_info_still[i].sample_num << " samples";
         }
          cursor_still->SeekToFirst();
		}
		
        // ok ,now start to read data
        int class_id = 0;
		//int batch_count = 0;
        //vector<bool>  is_class_finished(class_num_per_batch,false);
        vector<shared_ptr<QueuePair> > qps;
        try {
            // To ensure deterministic runs, only start running once all solvers
            // are ready. But solvers need to peek on one item during initialization,
            // so read one item, then wait for the next solver.
            for (int i = 0; i < solver_count; ++i) {
                shared_ptr<QueuePair> qp(new_queue_pairs_.pop());
                //bool flag = is_class_finished[class_id];
                read_one(class_db_reader_info[class_id].cursor.get(), qp.get(), class_db_reader_info[class_id].first_key, class_db_reader_info[class_id].current_key,
                         class_db_reader_info[class_id].last_key,frame_count[i]/*,flag*/);
                //is_class_finished[class_id] = flag;
                //class_db_reader_info[class_id].sample_count++;
				if(cross_model){
					read_one(class_db_reader_info_still[class_id].cursor.get(), qp.get(), class_db_reader_info_still[class_id].first_key, class_db_reader_info_still[class_id].current_key,
                         class_db_reader_info_still[class_id].last_key );
					frame_count[i]++;
				}
				
                qps.push_back(qp);
            }
            clip_count = 1;
            // LOG(INFO) << "here";
            // Main loop
            while (!must_stop()) {
                // already enough to fulfill a batch, reset variations to get ready for next batch
                if(clip_count > class_num_per_batch*clip_num_per_class){
                    class_id = 0;
                    clip_count = 0;
					//batch_count ++;
                    frame_count.clear();
                    frame_count.assign(solver_count,0);
                    /*bool is_finished = true;
                    for(int m = 0; m < is_class_finished.size(); m++){
                        is_finished &= is_class_finished[m];
                    }*/
                    //if(is_finished)
					//if(batch_count >= 5)
					{
                        shuffle(class_db_reader_info.begin(), class_db_reader_info.end());
						//batch_count = 0;
                        //is_class_finished.assign(class_num_per_batch,false);
                    }
                }
                else{
                    class_id = (class_id + 1) % class_num_per_batch;
                }
				std::string class_label = class_db_reader_info[class_id].first_key.substr(13, 4);//need to be adaptive
                int class_idx = atoi(class_label.c_str());
                for (int i = 0; i < clip_num_per_class; ++i) {
                    bool do_pad = (clip_count==clip_num_per_class*class_num_per_batch) ? true : false;
                    for(int j = 0; j < solver_count; ++j){
                        int tmp_size = (do_pad) ? (batch_size - frame_count[j]) : 0;
                        //bool flag = is_class_finished[class_id];
                        read_one(class_db_reader_info[class_id].cursor.get(), qps[j].get(), class_db_reader_info[class_id].first_key, class_db_reader_info[class_id].current_key,
                                 class_db_reader_info[class_id].last_key, tmp_size, /*flag,*/ do_pad);
                        //is_class_finished[class_id] = flag;
                        frame_count[j] += tmp_size;
						if(cross_model&&!do_pad){
					      read_one(class_db_reader_info_still[class_idx].cursor.get(), qps[j].get(), class_db_reader_info_still[class_idx].first_key, class_db_reader_info_still[class_idx].current_key,
                          class_db_reader_info_still[class_idx].last_key);
					      frame_count[j]++;
				        }
                    }
                    clip_count ++;
                    if (do_pad){//already fulfill the batch
                        break;
                    }
                }
                //clip_count += clip_num_per_class;
                // Check no additional readers have been created. This can happen if
                // more than one net is trained at a time per process, whether single
                // or multi solver. It might also happen if two data layers have same
                // name and same source.
                CHECK_EQ(new_queue_pairs_.size(), 0);
            }
        }
        catch (boost::thread_interrupted&) {
            // Interrupted exception is expected on shutdown
        }
    }
}

// read one clip from the pos of cursor
void DataReader::Body::read_one(db::Cursor* cursor, QueuePair* qp, int &clip_size, bool is_pad) {
    if (!is_pad){// normally read
        // first frame of the clip
        Datum* datum = qp->free_.pop();
        datum->ParseFromString(cursor->value());
        qp->full_.push(datum);

        int clip_label  = datum->label();
        clip_size = 1;
        while(true){
            cursor->Next();

            //the end of the whole DB
            if (!cursor->valid()) {
                DLOG(INFO) << "Restarting data prefetching from start.";
                cursor->SeekToFirst();
                break;
            }
            else{
                Datum *datum1 = qp->free_.pop();
                datum1->ParseFromString(cursor->value());
                int tmp_label = datum1->label();
                // the cont. of the clip
                if(clip_label == tmp_label){
                    qp->full_.push(datum1);
                    clip_size++;
                }
                else{// the begin of next clip
                    qp->free_.push(datum1);
                    break;
                }

            }
        }
		//qss
		/*fstream output;
		output.open("/home/data/qiaoshishi/test.txt",ios::out|ios::app);
		output<<clip_label<<"\t"<<clip_label%1000<<"\t"<<clip_label/1000<<"\t"<<clip_size<<std::endl;
		output.close();*/

    }
    else{// pad the batch
        if(clip_size > 0){
            for(int i = 0; i < clip_size; i++){
                Datum* datum = qp->free_.pop();
                datum->ParseFromString(cursor->value());
                datum->set_label(-1);
                qp->full_.push(datum);
            }
        }

    }
    // go to the next clip
}
void DataReader::Body::read_one(db::Cursor *cursor, QueuePair *qp){
  Datum* datum = qp->free_.pop();
  // TODO deserialize in-place instead of copy?
  datum->ParseFromString(cursor->value());
  qp->full_.push(datum);

  // go to the next iter
  cursor->Next();
  if (!cursor->valid()) {
    DLOG(INFO) << "Restarting data prefetching from start.";
    cursor->SeekToFirst();
  }
}
void DataReader::Body::read_one(db::Cursor *cursor, QueuePair *qp, string &first_key, string &current_key, string &last_key, int &clip_size,  /*bool &is_end,*/ bool is_pad){
    if (!is_pad){// normally read
        // first frame of the clip
        Datum* datum = qp->free_.pop();
        datum->ParseFromString(cursor->value());
        qp->full_.push(datum);
        current_key = cursor->key();// current_key means the last datum read in the full_ queue, not the key of the cursor (maybe not read yet)

        int clip_label  = datum->label();
        clip_size = 1;
        while(true){
            cursor->Next();

            //the end of the whole DB, i.e., the end of the last class
            if (!cursor->valid()) {
                retrieval_one(cursor, first_key);// jump to the begin of the class
                //is_end = true;
                break;
            }
            else{
                Datum *datum1 = qp->free_.pop();
                datum1->ParseFromString(cursor->value());
                int tmp_label = datum1->label();
                // the cont. of the clip
                if(clip_label == tmp_label){
                    qp->full_.push(datum1);
                    clip_size ++;
                    current_key = cursor->key();
                }
                else{// the begin of next clip of ?? class
                    if(current_key == last_key){// the end of the class
                        retrieval_one(cursor,first_key);
                        //current_key = first_key;
                        //is_end = true;
                    }
                    qp->free_.push(datum1);
                    break;
                }

            }
        }

    }
    else{// pad the batch
        if(clip_size > 0){
            for(int i = 0; i < clip_size; i++){
                Datum* datum = qp->free_.pop();
                datum->ParseFromString(cursor->value());
                datum->set_label(-1);
                qp->full_.push(datum);
            }
        }

    }
    // go to the next clip
}

void DataReader::Body::read_one(db::Cursor *cursor, QueuePair *qp, string &first_key, string &current_key, string &last_key){
        if (cursor->valid())
		{
			Datum* datum = qp->free_.pop();
			// TODO deserialize in-place instead of copy?
			datum->ParseFromString(cursor->value());
			qp->full_.push(datum);
			if (current_key == last_key)
			{
				retrieval_one(cursor, first_key);
				current_key = first_key;
			}
			else
			{
				cursor->Next();
				current_key = cursor->key();
			}
		}
		else
		{

			LOG(INFO) << "Cursor pos error " ;
		}	
}

void DataReader::Body::retrieval_one(db::Cursor *cursor, string &key_word) {//qss make the cursor  the position of key

    cursor->Retrieval(key_word);
    if (!cursor->valid())
    {
        LOG(INFO) << "DB doesn't have item with key " << key_word;
    }

}

}  // namespace caffe
