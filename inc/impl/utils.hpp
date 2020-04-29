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

// x[batch_size, 20, 5]
torch::Tensor xywh2xyxy(torch::Tensor x){
	torch::Tensor y = x;
	y.select(2, 1) = x.select(2, 1) - x.select(2, 3) / 2;
	y.select(2, 2) = x.select(2, 2) - x.select(2, 4) / 2;
	y.select(2, 3) = x.select(2, 1) + x.select(2, 3) / 2;
	y.select(2, 4) = x.select(2, 2) + x.select(2, 4) / 2;
	return y;
}

torch::Tensor bbox_iou(torch::Tensor box1, torch::Tensor box2){
	// Get the coordinates of bounding boxes
	  torch::Tensor b1_x1, b1_y1, b1_x2, b1_y2;
	  b1_x1 = box1.select (1, 0);
	  b1_y1 = box1.select (1, 1);
	  b1_x2 = box1.select (1, 2);
	  b1_y2 = box1.select (1, 3);
	  torch::Tensor b2_x1, b2_y1, b2_x2, b2_y2;
	  b2_x1 = box2.select (1, 0);
	  b2_y1 = box2.select (1, 1);
	  b2_x2 = box2.select (1, 2);
	  b2_y2 = box2.select (1, 3);

	  // et the corrdinates of the intersection rectangle

	  torch::Tensor inter_rect_x1 = torch::max (b1_x1, b2_x1);
	  torch::Tensor inter_rect_y1 = torch::max (b1_y1, b2_y1);
	  torch::Tensor inter_rect_x2 = torch::min (b1_x2, b2_x2);
	  torch::Tensor inter_rect_y2 = torch::min (b1_y2, b2_y2);

	  // Intersection area
	  torch::Tensor inter_area = torch::max (inter_rect_x2 - inter_rect_x1 + 1, torch::zeros (inter_rect_x2.sizes ()))
	      * torch::max (inter_rect_y2 - inter_rect_y1 + 1, torch::zeros (inter_rect_x2.sizes ()));

	  // Union Area
	  torch::Tensor b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1);
	  torch::Tensor b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1);

	  torch::Tensor iou = inter_area / (b1_area + b2_area - inter_area);

	  return iou;
}

struct myIndex{
	int batch_index;
	int t_box_index;
	myIndex(int a = 0, int b = 0):batch_index(a), t_box_index(b){ }
	bool operator==(const myIndex & i){
		return (batch_index == i.batch_index && t_box_index == i.t_box_index);
	}
};

bool boxDetected(std::vector<myIndex> box_indexes, myIndex index){
	for(auto i : box_indexes){
		if(index == i){
			return true;
		}
	}
	return false;
}

// outputs[batches_pred_num, 8] batch_index, left x, left y, right x, right y, object confidence, class_score, class_id
// targets[batch_size, 20, 5] class_id, left x, left y, right x, right y
torch::Tensor get_batch_statistics(torch::Tensor outputs, torch::Tensor targets, float iou_threshold){
	if(outputs.dim() == 1)
		return torch::tensor({0});
	torch::Tensor batch_metrics;/* = torch::zeros()*/
	int batches_pred_num = outputs.size(0);
	torch::Tensor pred_scores = outputs.select(1, 5);
	torch::Tensor pred_labels = outputs.select(1, 7);

	torch::Tensor true_positives = torch::zeros({1, batches_pred_num}, torch::kFloat);
	std::vector<myIndex> detected_boxes {};

	for(int i = 0; i < batches_pred_num; i++){
		torch::Tensor output = outputs[i];

		torch::Tensor pred_box = output.slice(0, 1, 5);
//		torch::Tensor pred_score = output.select(0, 5);
		torch::Tensor pred_label = output.select(0, 7);
		int batch_index = output.select(0, 0).item<int>();

		torch::Tensor target = targets[batch_index];

		torch::Tensor ious;
		for(int j = 0; j < target.size(0); j++){

			if(target[j].sum().item<float>() == 0.0)
				break;
			if(pred_label.item<int>() != target[j].select(0, 0).item<int>())
				continue;
			torch::Tensor iou = bbox_iou(pred_box.unsqueeze(0), target[j].slice(0, 1, 5).unsqueeze(0));
			if(j == 0){
				ious = iou;
			}
			else{
				ious = torch::cat({ious, iou}, 0);
			}
		}

		torch::Tensor best_iou, box_index;
		std::tie(best_iou, box_index) = ious.max(0);
		myIndex index(batch_index, box_index.item<int>());
		std::cout << best_iou.item<float>() << std::endl;
		if(best_iou.item<float>() > iou_threshold && !boxDetected(detected_boxes, index)){
			true_positives[0][i] = 1.0;
			detected_boxes.push_back(index);
		}
	}

	batch_metrics = torch::cat({true_positives, pred_scores.unsqueeze(0), pred_labels.unsqueeze(0)}, 0);

	std::cout << "got batch metrics" << std::endl;
	return batch_metrics;
}

float compute_ap(torch::Tensor r, torch::Tensor p){
	torch::Tensor mrec = torch::cat({torch::tensor({0.0}), r, torch::tensor({1.0})}, 0);
	torch::Tensor mpre = torch::cat({torch::tensor({0.0}), p, torch::tensor({1.0})}, 0);

	for(int i = mpre.size(0) - 1; i >= 0; i--){
		mpre[i - 1] = torch::max(mpre[i-1], mpre[i]);
	}
	torch::Tensor index = mrec.slice(0, 1, mrec.size(0) - 1) != mrec.slice(0, 0, mrec.size(0) - 2);
//	std::cout << index << std::endl;
	int idx = 0;
	for(int i = 0; i < index.size(0); i++){
		if(index[i].item<int>() == 1){
			idx = i;
			break;
		}
	}
//	std::cout << idx << std::endl;
	float ap = torch::sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]).item<float>();
//	std::cout << ap << std::endl;
	return ap;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,std::vector<int>> ap_per_class(torch::Tensor true_positives, torch::Tensor pred_scores, torch::Tensor pred_labels, std::vector<int> target_cls){
	std::sort(target_cls.begin(), target_cls.end());
	std::vector<int>::iterator it;
	it = std::unique(target_cls.begin(), target_cls.end());
	std::vector<int> labels = {target_cls.begin(), it};
	std::cout << labels << std::endl;
	torch::Tensor i = torch::argsort(-pred_scores);
//	std::cout << i << std::endl;
	torch::Tensor tp, conf, pred_cls;
	tp = true_positives.index(i);
	conf = pred_scores.index(i);
	pred_cls = pred_labels.index(i);
	torch::Tensor target_clst = torch::from_blob(target_cls.data(), {int(target_cls.size())}, torch::kInt);
//	std::cout << target_clst << std::endl;
//	std::cout << tp << std::endl;
	std::vector<float> ap, p, r;
	for(int c : labels){
		torch::Tensor tc = torch::tensor({c}, torch::kFloat);
		torch::Tensor cls_index = pred_cls == tc;
//		std::cout << cls_index << std::endl;
		torch::Tensor n_gt = (target_clst == tc).sum();
//		std::cout << n_gt << std::endl;
		torch::Tensor n_p = cls_index.sum();
		if(n_p.item<int>() == 0 && n_gt.item<int>() == 0){
			continue;
		}
		else if(n_p.item<int>() == 0 || n_gt.item<int>() == 0){
			ap.push_back(0.0);
			p.push_back(0.0);
			r.push_back(0.0);
		}
		else{
			torch::Tensor fpc = (1 - tp.index(cls_index)).cumsum(0);
//			std::cout << fpc << std::endl;
			torch::Tensor tpc = tp.index(cls_index).cumsum(0);
//			std::cout << tpc << std::endl;
			torch::Tensor recall_curve = tpc / (n_gt + 1e-16);
//			std::cout << recall_curve << std::endl;
//			std::cout << recall_curve[-1].item<float>() << std::endl;
//			std::cout << recall_curve[recall_curve.size(0) - 1].item<float>() << std::endl;
			r.push_back(recall_curve[-1].item<float>());
			torch::Tensor precision_curve = tpc / (tpc + fpc);
//			std::cout << precision_curve << std::endl;
//			std::cout << precision_curve[-1].item<float>() << std::endl;
			p.push_back(precision_curve[-1].item<float>());
			ap.push_back(compute_ap(recall_curve, precision_curve));
		}
	}
	torch::Tensor ap_t, p_t, r_t;
	std::cout << "ap: " << ap << std::endl;
	std::cout << sizeof(float) << std::endl;
	ap_t = torch::from_blob(ap.data(), {int(ap.size())}, torch::kFloat16);
	p_t = torch::from_blob(p.data(), {int(p.size())}, torch::kFloat);
	r_t = torch::from_blob(r.data(), {int(r.size())}, torch::kFloat);
	torch::Tensor f1 = 2 * p_t * r_t / (p_t + r_t + 1e-16);
//	std::cout << f1 << std::endl;
	return {p_t, r_t, ap_t, f1, labels};
}

#endif
