#pragma once
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <sstream>

using Data = std::vector<std::pair<std::string, std::vector<std::vector<float>>>>;

struct yoloOps {
	int image_size = 416;
	int batch_size = 2;
	int num_classes = 2;
	std::string cfg_path =
			"/path/to/your/project/cfg/yolov3.cfg";
	std::string train_weights_path =
			"/path/to/your/project/weights/darknet53.conv.74";
	float learn_rate = 0.001;  //defalt: 1e-3, 0.001 in cfg
//    float weight_decay = 0.0005;
	int iterations = 100;
	int log_interval = 10;
	int evaluation_interval = 1;
};

static yoloOps yolo_ops;

struct voc2clsOptions {
	std::string train_path =
			"/path/to/your/dataset/VOCdevkit/VOC2012/2cls/train.txt";
	std::string val_path =
			"/path/to/your/dataset/VOCdevkit/VOC2012/2cls/val.txt";
	std::string img_path =
			"/path/to/your/dataset/VOCdevkit/VOC2012/JPEGImages/";
	std::string trans_label_path =
			"/path/to/your/dataset/VOCdevkit/VOC2012/2cls/2clsLabels/";
};

static voc2clsOptions voc2_ops;

std::pair<Data, Data>
readInfo();

std::tuple<cv::Mat, std::vector<int>>
pad2square(cv::Mat img);

class CustomDataset: public torch::data::datasets::Dataset<CustomDataset> {
	using Example = torch::data::Example<>;

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

torch::Tensor
bbox_wh_iou(torch::Tensor wh1, torch::Tensor wh2);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
		torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
		torch::Tensor, torch::Tensor>
buildTargets(torch::Tensor targets, torch::Tensor pred_boxes,
		torch::Tensor pred_cls, torch::Tensor scaled_anchors,
		float ignore_thres);

#include <impl/utils.hpp>
