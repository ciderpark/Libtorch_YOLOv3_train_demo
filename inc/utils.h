#pragma once
#include <iostream>
#include <sstream>
#include <algorithm>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>

using Data = std::vector<std::pair<std::string, std::vector<std::vector<float>>>>;
using Example = torch::data::Example<>;

struct yoloOps {
	int image_size = 416;
	int batch_size = 2;
	int num_classes = 2;
	std::string cfg_path =
			"/your/path/to/cfg/yolov3.cfg";
	std::string train_weights_path =
			"/your/path/to/weights/darknet53.conv.74";
	float learn_rate = 1e-5;  //defalt: 1e-3, 0.001 in cfg
//    float weight_decay = 0.0005;
	int iterations = 100;
	int log_interval = 10;
	int evaluation_interval = 1;
};

static yoloOps yolo_ops;

struct voc2clsOptions {
	std::string train_path =
			"/your/path/to/train.txt";
	std::string val_path =
			"/your/path/to/val.txt";
	std::string img_path =
			"/your/path/to/VOCdevkit/VOC2012/JPEGImages/";
	std::string trans_label_path =
			"/your/path/to/VOCdevkit/VOC2012/2cls/2clsLabels/";
	std::vector<std::string> classes_name {"aeroplane", "bicycle"};
};

static voc2clsOptions voc2_ops;

std::pair<Data, Data>
readInfo();

std::tuple<cv::Mat, std::vector<int>>
pad2square(cv::Mat img);

class CustomDataset: public torch::data::datasets::Dataset<CustomDataset> {
//	using Example = torch::data::Example<>;

	Data data;

public:
	CustomDataset(const Data &data) :
			data(data) {
	}
	Example
	get(size_t index);

	torch::optional<size_t> size() const {
		return data.size();
	}
};

template<typename T = Example>
struct StackAndCat;

template<>
struct StackAndCat<Example> : public torch::data::transforms::Collation<Example> {
	Example apply_batch(std::vector<Example> examples) override {
		std::vector<torch::Tensor> data, targets;
		data.reserve(examples.size());
		targets.reserve(examples.size());
		for (int64_t i = 0; i < examples.size(); i++) {
			examples[i].target.select(1, 0) = i;
			data.push_back(std::move(examples[i].data));
			targets.push_back(std::move(examples[i].target));
		}
		return {torch::stack(data), torch::cat(targets)};
	}
};

torch::Tensor
bbox_wh_iou(torch::Tensor wh1, torch::Tensor wh2);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
		torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
		torch::Tensor, torch::Tensor>
buildTargets(torch::Tensor targets, torch::Tensor pred_boxes,
		torch::Tensor pred_cls, torch::Tensor scaled_anchors,
		float ignore_thres);

torch::Tensor xywh2xyxy(torch::Tensor);

torch::Tensor bbox_iou(torch::Tensor box1, torch::Tensor box2, bool xyxy);

torch::Tensor get_batch_statistics(torch::Tensor outputs, torch::Tensor targets,
		float iou_threshold);

torch::Tensor compute_ap(torch::Tensor r, torch::Tensor p);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
		torch::Tensor> ap_per_class(torch::Tensor true_positives,
		torch::Tensor pred_scores, torch::Tensor pred_labels, torch::Tensor);

#include <impl/utils.hpp>
