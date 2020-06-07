#include <iostream>
#include <chrono>
#include <time.h>
#include <iomanip>

#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "yolov3.h"
#include "utils.h"

using namespace std;
using namespace std::chrono;

void printLog(){
	std::cout << std::setw(15) << std::setiosflags(std::ios::left) << "| Metrics"
				<< std::setw(15) << std::setiosflags(std::ios::left) << "| YOLO Layer 0"
				<< std::setw(15) << std::setiosflags(std::ios::left) << "| YOLO Layer 1"
				<< std::setw(15) << std::setiosflags(std::ios::left) << "| YOLO Layer 2"
				<< "|"
				<< std::endl;
	vector<string> log_item {"grid_size","loss","x","y","w","h","conf", "cls", "cls_acc", "recall50", "recall75", "precision", "conf_obj", "conf_noobj"};
	for(string s : log_item){
		std::cout << std::setw(15) << std::setiosflags(std::ios::left) << "| " + s;
		for(int i = 0; i < metrics_v.size(); i++){
			std::cout << std::setw(15) << std::setiosflags(std::ios::left) << "| " + to_string(metrics_v[i][s]);
		}
		std::cout << "|" << std::endl;
	}
	std::cout << "Total Loss: " << metrics_v[0]["loss"] + metrics_v[1]["loss"] + metrics_v[2]["loss"] << "\n" << std::endl;
	return;
}

template <typename DataLoader>
void train(shared_ptr<Darknet> net, DataLoader& loader, torch::optim::Optimizer& optimizer,  size_t epoch, size_t train_size, torch::Device device){
	net->train ();
	size_t index = 0;
	float Loss = 0.0;
	for(auto& batch : loader){
		torch::Tensor input = batch.data.to (device);
		torch::Tensor targets = batch.target.to (device);
		torch::Tensor output, loss;
		optimizer.zero_grad ();
		std::tie (output, loss) = net->forward (input, targets);

		loss.backward ();
		optimizer.step ();
		Loss += loss.item<float> ();
	    if (++index % yolo_ops.log_interval == 0)
	    {
	      auto end = std::min (train_size, (index/* + 1*/) * yolo_ops.batch_size);
	      std::cout << " -- Train Epoch: " << epoch << "/" << yolo_ops.iterations << " -- Batch: " << end << "/" << train_size << "\t -- Total Loss Avg: " << Loss / end << std::endl;
	      printLog();
	    }
//		return;
	}
}

template <typename DataLoader>
void evaluate(shared_ptr<Darknet> net, DataLoader& loader, size_t valid_size, float iou_thres, float conf_thres, float nms_thres, torch::Device device){
	net->eval();
	torch::NoGradGuard no_grad;
	std::cout << "evaluating model ..." << std::endl;
	vector<torch::Tensor> sample_metrics {};
	vector<torch::Tensor> labels {};
	for(auto& batch : loader){
		torch::Tensor input = batch.data.to (device);
		torch::Tensor targets = batch.target.to (device);
		labels.push_back(targets.select(1, 1));
		targets.slice(1, 2, 6) = xywh2xyxy(targets.slice(1, 2, 6)) * yolo_ops.image_size;
		torch::Tensor output = net->forward(input);
		torch::Tensor results = net->write_results(output, yolo_ops.num_classes, conf_thres, nms_thres).to(device);
		if(results.dim() > 1){
			sample_metrics.push_back(get_batch_statistics(results, targets, iou_thres));
		}
	}
	torch::Tensor labels_t = torch::cat(labels);
	torch::Tensor sample_metrics_t = torch::cat(sample_metrics);
	torch::Tensor true_positives = sample_metrics_t.select(1, 0);
	torch::Tensor pred_scores = sample_metrics_t.select(1, 1);
	torch::Tensor pred_labels = sample_metrics_t.select(1, 2);
	torch::Tensor precision, recall, AP, f1, ap_class;
	std::tie(precision, recall, AP, f1, ap_class) = ap_per_class(true_positives, pred_scores, pred_labels, labels_t);
	std::cout << "Average Precisions:" << std::endl;
	for(int i = 0; i < ap_class.size(0); i++){
		int c = ap_class[i].item<int>();
		std::cout << "Class: " << std::setw(15) << std::setiosflags(std::ios::left) << voc2_ops.classes_name[c] << " - AP: " << AP[c].item<float>() << std::endl;
	}
	std::cout << "mAP: " << AP.mean().item<float>() << std::endl;
}

int
main (int argc,
      char** argv)
{
  // Usage
  /*if (argc != 2)
   {
   std::cerr << "usage: yolov3-train <image path>\n";
   return -1;
   }*/
  // Device
  torch::DeviceType device_type;
  if (torch::cuda::is_available ())
  {
    device_type = torch::kCUDA;
  }
  else
  {
    device_type = torch::kCPU;
  }
  torch::Device device (device_type);
  // Dataset
  std::cout << "loading dataset ..." << std::endl;
  auto data = readInfo ();
  auto train_set = CustomDataset (data.first).map (StackAndCat<>());
  auto train_size = train_set.size ().value ();
  auto train_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler> (std::move (train_set), yolo_ops.batch_size);

  auto valid_set = CustomDataset (data.second).map (StackAndCat<>());
  auto valid_size = valid_set.size ().value ();
  auto valid_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler> (std::move (valid_set), yolo_ops.batch_size);
  std::cout << "Done." << std::endl;

  // Net
  std::cout << "loading cfg & creating modules ..." << std::endl;
  auto net = std::make_shared<Darknet> (yolo_ops.cfg_path, &device);
  map<string, string> *info = net->get_net_info ();
  info->operator[] ("height") = std::to_string (yolo_ops.image_size);
  std::cout << "Done." << std::endl;
  // Load weights
  std::cout << "loading weight ..." << endl;
//  net->load_weights (yolo_ops.train_weights_path);
  torch::load(net, yolo_ops.train_weights_path);
  std::cout << "Done." << endl;

  net->to (device);
  torch::optim::Adam optimizer (net->parameters (), torch::optim::AdamOptions (yolo_ops.learn_rate)/*.weight_decay (yolo_ops.weight_decay)*/);

  std::cout << "train start ..." << std::endl;
  for (size_t i = 0; i < yolo_ops.iterations; i++)
  {
	train(net, *train_loader, optimizer, i + 1, train_size, device);
	if(!((i + 1) % yolo_ops.evaluation_interval)){
		evaluate(net, *valid_loader, valid_size, 0.5, 0.6, 0.4, device);
	}
//	break;
  }
  torch::save (net, "yolov3-voc2t.pt");
  std::cout << "train done." << std::endl;
  return (0);
}
