#pragma once

#include <torch/torch.h>
#include <string>
#include <vector>
#include <map>
#include "utils.h"

using namespace std;

map<string, float> metrics = { { "loss", 0.0 }, { "x", 0.0 }, { "y", 0.0 }, {
		"w", 0.0 }, { "h", 0.0 }, { "conf", 0.0 }, { "cls", 0.0 }, { "cls_acc",
		0.0 }, { "recall50", 0.0 }, { "recall75", 0.0 }, { "precision", 0.0 }, {
		"conf_obj", 0.0 }, { "conf_noobj", 0.0 }, { "grid_size", 0.0 } };

vector<map<string, float>> metrics_v(3);

struct Darknet: torch::nn::Module {

public:

	Darknet(const string conf_file, torch::Device *device);

	map<string, string>*
	get_net_info();

	void
	load_weights(const string weight_file);

	torch::Tensor
	forward(torch::Tensor x);

	std::tuple<torch::Tensor, torch::Tensor>
	forward(torch::Tensor x, torch::Tensor target);

	torch::Tensor
	write_results(torch::Tensor prediction, int num_classes, float confidence,
			float nms_conf = 0.4);

private:

	torch::Device *_device;

	vector<map<string, string>> blocks;

	torch::nn::Sequential features;

	vector<torch::nn::Sequential> module_list;

	// load YOLOv3
	void
	load_cfg(const string cfg_file);

	void
	create_modules();

	int
	get_int_from_cfg(map<string, string> block, string key, int default_value);

	string
	get_string_from_cfg(map<string, string> block, string key,
			string default_value);
};
#include <impl/yolov3.hpp>
