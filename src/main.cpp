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
	}
}

template <typename DataLoader>
void evaluate(shared_ptr<Darknet> net, DataLoader& loader, size_t valid_size, torch::Device device){
	net->eval();
	torch::NoGradGuard no_grad;
	std::cout << "evaluating model ..." << std::endl;
	vector<torch::Tensor> sample_metrics {};
	vector<torch::Tensor> targets_v {};
	for(auto& batch : loader){
		torch::Tensor input = batch.data.to (device);
		torch::Tensor targets = batch.target/*.to (device)*/;
		targets = xywh2xyxy(targets);
		targets.slice(2, 1, 5) *= yolo_ops.image_size;
		torch::Tensor output = net->forward(input);
		/*//
				std::cout << "output.sizes (): " << output.sizes() << std::endl;
				//    std::cout << output.select(2, 4).sizes() << std::endl;
					float conf = 0.0;
					for (int i = 0; i < output.size(1); i++) {
						conf = output[0][i][4].item<float>();
						if (conf > 0.0) {
				//    std::cout << "confidence: " << conf_mask[0][i][0] << ", index: " << i << std::endl;
							float max = 0.0;
							int idx = 0;
							for (int j = 5; j < output.size(2); j++) {
								if (output[0][i][j].item<float>() > max) {
									max = output[0][i][j].item<float>();
									idx = j;
								}
							}
				//    std::cout << "pre: " << output[0][i].max() << ", index: " << i << std::endl;
							float confi = output[0][i][4].item<float>();
							std::cout << "class_id: " << idx - 5 << ", pre_index: " << i
									<< ", conf: " << confi << ", class_score: " << max
									<< ", x:" << output[0][i][0].item<float>() << ", y: " << output[0][i][1].item<float>()
									<< ", w: " << output[0][i][2].item<float>() << ", h: " << output[0][i][3].item<float>()
									<< std::endl;
						}
					}
				//*/
		output = net->write_results(output, yolo_ops.num_classes, 0.6, 0.4);
		std::cout << output << std::endl;
//		output = output.to(device);
//		std::cout << output.device() << std::endl;
		sample_metrics.push_back(get_batch_statistics(output, targets, 0.2));
		targets_v.push_back(targets);
	}
	std::cout << "here" << std::endl;
	torch::Tensor true_positives, pred_scores, pred_labels;
	bool write = false;
	for(int i = 0; i < sample_metrics.size(); i++){
		if(sample_metrics[i].dim() == 1)
			continue;
		torch::Tensor p = sample_metrics[i].select(0, 0);
		torch::Tensor s = sample_metrics[i].select(0, 1);
		torch::Tensor l = sample_metrics[i].select(0, 2);
		if(!write){
			true_positives = p;
			pred_scores = s;
			pred_labels = l;
			write = true;
		}
		else{
			true_positives = torch::cat({true_positives, p}, 0);
			pred_scores = torch::cat({pred_scores, p}, 0);
			pred_labels = torch::cat({pred_labels, p}, 0);
		}
	}
	vector<int> labels {};
	for(torch::Tensor t : targets_v){
		for(int i = 0; i < t.size(0); i++){
			for(int j = 0; j < t.size(1); j++){
				if(t[i][j].sum().item<float>() == 0.0)
					continue;
				labels.push_back(t[i][j].select(0, 0).item<int>());
			}
		}
	}
	std::cout << true_positives.sizes() << pred_scores.sizes() << pred_labels.sizes() << std::endl;
	std::cout << labels.size() << std::endl;
	torch::Tensor precision, recall, AP, f1;
	std::vector<int> ap_class;
	std::tie(precision, recall, AP, f1, ap_class) = ap_per_class(true_positives, pred_scores, pred_labels, labels);
	std::cout << precision << recall << AP << f1 << ap_class << std::endl;
	std::cout << "evaluation done." << std::endl;
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
  auto train_set = CustomDataset (data.first).map (torch::data::transforms::Stack<> ());
  auto train_size = train_set.size ().value ();
  auto train_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler> (std::move (train_set), yolo_ops.batch_size);

  auto valid_set = CustomDataset (data.second).map (torch::data::transforms::Stack<> ());
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
//	train(net, *train_loader, optimizer, i + 1, train_size, device);
	if(!((i + 1) % yolo_ops.evaluation_interval)){
		evaluate(net, *train_loader, train_size, device);
	}
	break;
  }
//  torch::save (net, "yolov3-voc2t.pt");
  std::cout << "train done." << std::endl;
  return (0);
}
