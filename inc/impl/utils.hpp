#ifndef UTILS_HPP_
#define UTILS_HPP_
#include "utils.h"

std::pair<Data, Data> readInfo() {
	Data train, valid;

	std::string name;
	std::vector<std::vector<float>> label;

	//train set
	std::ifstream stream_t(voc2_ops.train_path);
	assert(stream_t.is_open());
	while (true) {
		stream_t >> name;
		if (stream_t.eof())
			break;
		std::string path = voc2_ops.img_path + name + ".jpg";
		std::string path_l = voc2_ops.trans_label_path + name + ".txt";

		std::ifstream stream_l(path_l);
		assert(stream_l.is_open());
		/*if (!stream_l.is_open())
		 continue;*/
		//
		float c, x, y, w, h;
		while (true) {
			stream_l >> c >> x >> y >> w >> h;
			if (stream_l.eof())
				break;
			label.push_back( { c, x, y, w, h });
		}

		train.push_back(std::make_pair(path, label));
		label.clear();
	}

	//valid set
	std::ifstream stream_v(voc2_ops.val_path);
	assert(stream_v.is_open());
	while (true) {
		stream_v >> name;
		if (stream_v.eof())
			break;

		std::string path = voc2_ops.img_path + name + ".jpg";
		std::string path_l = voc2_ops.trans_label_path + name + ".txt";

		std::ifstream stream_l(path_l);
		assert(stream_l.is_open());
		/*if (!stream_l.is_open())
		 continue;*/
		//
		float c, x, y, w, h;
		while (true) {
			stream_l >> c >> x >> y >> w >> h;
			if (stream_l.eof())
				break;
			label.push_back( { c, x, y, w, h });
		}

		valid.push_back(std::make_pair(path, label));
		label.clear();
	}

	std::random_shuffle(train.begin(), train.end());
	std::random_shuffle(valid.begin(), valid.end());
	return std::make_pair(train, valid);
}

std::tuple<cv::Mat, std::vector<int>> pad2square(cv::Mat img) {
	int w = img.cols;
	int h = img.rows;
	int diff = std::abs(w - h);
	int top = 0, bottom = 0, left = 0, right = 0;
	if (w > h) {
		top = diff / 2;
		bottom = diff - top;
	} else {
		left = diff / 2;
		right = diff - left;
	}
	cv::Mat rst;
	cv::copyMakeBorder(img, rst, top, bottom, left, right, cv::BORDER_CONSTANT,
			0);
	std::vector<int> pad { top, bottom, left, right };
	return {rst, pad};
}

using Example = torch::data::Example<>;

Example CustomDataset::get(size_t index) {
	std::string path = data[index].first;
	auto mat = cv::imread(path);
	assert(!mat.empty());

	int w_factor = mat.cols;
	int h_factor = mat.rows;
	cv::Mat c_mat;
	cv::cvtColor(mat, c_mat, cv::COLOR_BGR2RGB);

	cv::Mat s_mat;
	std::vector<int> pad;
	std::tie(s_mat, pad) = pad2square(c_mat);

	int w_padded = s_mat.cols;
	int h_padded = s_mat.rows;

	cv::resize(s_mat, s_mat,
			cv::Size(yolo_ops.image_size, yolo_ops.image_size));

	cv::Mat img_float;
	s_mat.convertTo(img_float, CV_32F, 1.0 / 255);
	torch::Tensor tdata = torch::from_blob(img_float.data, {
			yolo_ops.image_size, yolo_ops.image_size, 3 }, torch::kFloat);
	tdata = tdata.permute( { 2, 0, 1 }).contiguous();

	// label
	torch::Tensor tlabel = torch::zeros( { 20, 5 }, torch::kFloat);
	int idx = 0;
	for (auto l : data[index].second) {
		float x0 = w_factor * (l[1] - l[3] / 2);
		float y0 = h_factor * (l[2] - l[4] / 2);
		float x1 = w_factor * (l[1] + l[3] / 2);
		float y1 = h_factor * (l[2] + l[4] / 2);
		x0 += pad[2];
		y0 += pad[0];
		x1 += pad[3];
		y1 += pad[1];
		l[1] = ((x0 + x1) / 2) / w_padded;
		l[2] = ((y0 + y1) / 2) / h_padded;
		l[3] *= w_factor / float(w_padded);
		l[4] *= h_factor / float(h_padded);

		torch::Tensor label = torch::from_blob(l.data(), { 1, 5 },
				torch::kFloat);
		tlabel.slice(0, idx, idx + 1) = label;
		idx++;
	}

	return
	{	tdata, tlabel};
}

torch::Tensor bbox_wh_iou(torch::Tensor wh1, torch::Tensor wh2) {
	wh2 = wh2.t();
	torch::Tensor w1 = wh1[0], h1 = wh1[1];
	torch::Tensor w2 = wh2[0], h2 = wh2[1];
	torch::Tensor inter_area = torch::min(w1, w2) * torch::min(h1, h2);
	torch::Tensor union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area;
	return inter_area / union_area;
}

// targets[batch_size, 20 , 5]
// pred_boxs[batch_size, anchors_num, grid_size, grid_size, 4]
// pred_cls[batch_size, anchors_num, grid_size, grid_size, 80]
// scaled_anchors[3, 2]
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
		torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
		torch::Tensor, torch::Tensor> buildTargets(torch::Tensor targets,
		torch::Tensor pred_boxes, torch::Tensor pred_cls,
		torch::Tensor scaled_anchors, float ignore_thres) {
	int batch_size = pred_boxes.size(0);
	int anchors_num = pred_boxes.size(1);
	int class_num = pred_cls.size(-1);
	int grid_size = pred_boxes.size(2);
	torch::Device device = targets.device();
	// Output tensors
	torch::Tensor obj_mask = torch::zeros( { batch_size, anchors_num, grid_size,
			grid_size }, torch::kByte).to(device);
	torch::Tensor noobj_mask = torch::ones( { batch_size, anchors_num,
			grid_size, grid_size }, torch::kByte).to(device);
	torch::Tensor class_mask = torch::zeros( { batch_size, anchors_num,
			grid_size, grid_size }, torch::kFloat).to(device);
	torch::Tensor iou_scores = torch::zeros( { batch_size, anchors_num,
			grid_size, grid_size }, torch::kFloat).to(device);
	torch::Tensor tx = torch::zeros( { batch_size, anchors_num, grid_size,
			grid_size }, torch::kFloat).to(device);
	torch::Tensor ty = torch::zeros( { batch_size, anchors_num, grid_size,
			grid_size }, torch::kFloat).to(device);
	torch::Tensor tw = torch::zeros( { batch_size, anchors_num, grid_size,
			grid_size }, torch::kFloat).to(device);
	torch::Tensor th = torch::zeros( { batch_size, anchors_num, grid_size,
			grid_size }, torch::kFloat).to(device);
	torch::Tensor tcls = torch::zeros( { batch_size, anchors_num, grid_size,
			grid_size, class_num }, torch::kFloat).to(device);

	for (int i = 0; i < batch_size; i++) {
		for (int j = 0; j < targets.size(1); j++) {
			if (targets[i][j].sum().item<float>() == 0.0)
				break;

			// Convert to position relative to box
			torch::Tensor gx, gy, gwh;
			gx = targets[i][j][1] * grid_size;
			gy = targets[i][j][2] * grid_size;
			gwh = targets[i][j].slice(0, 3, 5) * grid_size;

			// Get grid box indices
			int gi = gx.item<int>(), gj = gy.item<int>();

			// Get anchors with best iou
			torch::Tensor iou0, iou1, iou2, ious;
			iou0 = bbox_wh_iou(scaled_anchors[0], gwh);
			iou1 = bbox_wh_iou(scaled_anchors[1], gwh);
			iou2 = bbox_wh_iou(scaled_anchors[2], gwh);

			ious = torch::stack( { iou0, iou1, iou2 }, 0);

			// Get anchors with best iou
			torch::Tensor best_ious, best_n;
			std::tie(best_ious, best_n) = ious.max(0);

			// Set masks
			obj_mask[i][best_n][gj][gi] = 1;
			noobj_mask[i][best_n][gj][gi] = 0;

			// Set noobj mask to zero where iou exceeds ignore threshold
			for (int na = 0; na < anchors_num; na++) {
				if (ious[na].item<float>() > ignore_thres)
					noobj_mask[i][na][gj][gi] = 0;
			}

			// Coordinates
			tx[i][best_n][gj][gi] = gx - gx.floor();
			ty[i][best_n][gj][gi] = gy - gy.floor();
			// Width and height
			tw[i][best_n][gj][gi] = torch::log(
					gwh[0] / scaled_anchors[best_n][0] + 1e-16);

			th[i][best_n][gj][gi] = torch::log(
					gwh[1] / scaled_anchors[best_n][1] + 1e-16);
			// One-hot encoding of label
			torch::Tensor target_label = targets[i][j][0];
			target_label = target_label.to(torch::kLong);
			tcls[i][best_n][gj][gi][target_label] = 1;
			// Compute label correctness and iou at best anchor
			class_mask[i][best_n][gj][gi] = (pred_cls[i][best_n][gj][gi].argmax(
					-1) == target_label).to(torch::kFloat);
			iou_scores[i][best_n][gj][gi] = bbox_wh_iou(
					pred_boxes[i][best_n][gj][gi].slice(0, 2, 4), gwh);
		}
	}
	torch::Tensor tconf = obj_mask.to(torch::kFloat).to(device);
	return
	{	iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf};
}

#endif
