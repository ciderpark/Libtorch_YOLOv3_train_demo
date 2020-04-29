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
  net->load_weights (yolo_ops.train_weights_path);
//  torch::load(net, yolo_ops.train_weights_path);
  std::cout << "Done." << endl;

  net->to (device);
  torch::optim::Adam optimizer (net->parameters (), torch::optim::AdamOptions (yolo_ops.learn_rate)/*.weight_decay (yolo_ops.weight_decay)*/);

  std::cout << "train start ..." << std::endl;
  for (size_t i = 0; i < yolo_ops.iterations; i++)
  {
	train(net, *train_loader, optimizer, i + 1, train_size, device);
  }
  torch::save (net, "yolov3-voc2t.pt");
  std::cout << "train done." << std::endl;
  return (0);
}
