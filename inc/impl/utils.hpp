#ifndef UTILS_HPP_
#define UTILS_HPP_
#include "utils.h"

void COCODateOps::loadClsName(){
	std::ifstream stream_t(classes_names_path);
	assert(stream_t.is_open());
	while(true){
		std::string name;
		stream_t >> name;
		if (stream_t.eof())
			break;
		classes_names.push_back(name);
	}
	stream_t.close();
}

std::pair<Data, Data> readInfo() {
	Data train, valid;

	std::string name;
	std::vector<std::vector<float>> label;

	//train set
	std::ifstream stream_t(data_ops.train_pathes);

	assert(stream_t.is_open());
	while (true) {
		stream_t >> name;
		if (stream_t.eof())
			break;
		std::string path = name;
		std::string path_l = name.replace (57, 6, "labels").replace (name.end () - 3, name.end (), "txt");
		std::ifstream stream_l(path_l);
		assert(stream_l.is_open());
		if (!stream_l.is_open())
		 continue;
		//
		float c, x, y, w, h, b = 0.0;
		while (true) {
			stream_l >> c >> x >> y >> w >> h;
			if (stream_l.eof())
				break;
			label.push_back( { b, c, x, y, w, h });
		}

		train.push_back(std::make_pair(path, label));
		label.clear();
	}

	//valid set
	std::ifstream stream_v(data_ops.valid_pathes);
	assert(stream_v.is_open());
	while (true) {
		stream_v >> name;
		if (stream_v.eof())
			break;

		std::string path = name;
		std::string path_l = name.replace (57, 6, "labels").replace (name.end () - 3, name.end (), "txt");

		std::ifstream stream_l(path_l);
//		assert(stream_l.is_open());
		if (!stream_l.is_open())
		 continue;
		//
		float c, x, y, w, h, b = 0.0;
		while (true) {
			stream_l >> c >> x >> y >> w >> h;
			if (stream_l.eof())
				break;
			label.push_back( { b, c, x, y, w, h });
		}

		valid.push_back(std::make_pair(path, label));
		label.clear();
	}
	std::cout << valid.size() << std::endl;
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
	torch::Tensor tlabel;
	std::vector < torch::Tensor > labels;

	for (auto l : data[index].second) {
		float x0 = w_factor * (l[2] - l[4] / 2);
		float y0 = h_factor * (l[3] - l[5] / 2);
		float x1 = w_factor * (l[2] + l[4] / 2);
		float y1 = h_factor * (l[3] + l[5] / 2);
		x0 += pad[2];
		y0 += pad[0];
		x1 += pad[3];
		y1 += pad[1];
		l[2] = ((x0 + x1) / 2) / w_padded;
		l[3] = ((y0 + y1) / 2) / h_padded;
		l[4] *= w_factor / float(w_padded);
		l[5] *= h_factor / float(h_padded);

//		std::cout << "l:\n" << l << std::endl;
		labels.push_back(
				torch::from_blob(l.data(), { 1, 6 }, torch::kFloat32).clone());
	}
	tlabel = torch::cat(labels);
//	std::cout << "tlabel:\n" << tlabel << std::endl;

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

// targets[num_labels, 6]
// pred_boxes[batch_size, anchors_num, grid_size, grid_size, 4]
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

	for (int i = 0; i < targets.size(0); i++) {
		// Convert to position relative to box
		torch::Tensor target_box = targets[i].slice(0, 2, 6) * grid_size;
		torch::Tensor gxy = target_box.slice(0, 0, 2);
		torch::Tensor gwh = target_box.slice(0, 2, 4);
		// Get anchors with best iou
		std::vector < torch::Tensor > iou;
		torch::Tensor ious;
		for (int i = 0; i < anchors_num; i++) {
			iou.push_back(bbox_wh_iou(scaled_anchors[i], gwh));
		}
		ious = torch::stack(iou);
		torch::Tensor best_iou, best_n;
		std::tie(best_iou, best_n) = ious.max(0);

		// Separate target values
		torch::Tensor b = targets[i].select(0, 0).to(torch::kLong);
		torch::Tensor target_label = targets[i].select(0, 1).to(torch::kLong);
		torch::Tensor gx = gxy.select(0, 0);
		torch::Tensor gy = gxy.select(0, 1);
		torch::Tensor gw = gwh.select(0, 0);
		torch::Tensor gh = gwh.select(0, 1);
		torch::Tensor gi = gx.to(torch::kLong);
		torch::Tensor gj = gy.to(torch::kLong);
		// Set masks
		obj_mask[b][best_n][gj][gi] = 1;
		noobj_mask[b][best_n][gj][gi] = 0;
		// Set noobj mask to zero where iou exceeds ignore threshold
		for (int na = 0; na < anchors_num; na++) {
			if (ious[na].item<float>() > ignore_thres)
				noobj_mask[b][na][gj][gi] = 0;
		}
		// Coordinates
		tx[b][best_n][gj][gi] = gx - gx.floor();
		ty[b][best_n][gj][gi] = gy - gy.floor();
		// Width and height
		tw[b][best_n][gj][gi] = torch::log(
				gw / scaled_anchors[best_n][0] + 1e-16);

		th[b][best_n][gj][gi] = torch::log(
				gh / scaled_anchors[best_n][1] + 1e-16);
		// One-hot encoding of label
		tcls[b][best_n][gj][gi][target_label] = 1;
		// Compute label correctness and iou at best anchor
		class_mask[b][best_n][gj][gi] = (pred_cls[b][best_n][gj][gi].argmax(-1)
				== target_label).to(torch::kFloat);

		iou_scores[b][best_n][gj][gi] = bbox_iou(
				pred_boxes[b][best_n][gj][gi].unsqueeze(0),
				target_box.unsqueeze(0), false).squeeze(0);
	}
	torch::Tensor tconf = obj_mask.to(torch::kFloat).to(device);
	return
	{	iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf};
}

torch::Tensor xywh2xyxy(torch::Tensor x) {
	torch::Tensor y = torch::zeros(x.sizes(), torch::kFloat);
	y.select(-1, 0) = x.select(-1, 0) - x.select(-1, 2) / 2;
	y.select(-1, 1) = x.select(-1, 1) - x.select(-1, 3) / 2;
	y.select(-1, 2) = x.select(-1, 0) + x.select(-1, 2) / 2;
	y.select(-1, 3) = x.select(-1, 1) + x.select(-1, 3) / 2;
	return y;
}

torch::Tensor bbox_iou(torch::Tensor box1, torch::Tensor box2,
		bool xyxy = true) {
	torch::Device device = box1.device();
	// Get the coordinates of bounding boxes
	torch::Tensor b1_x1, b1_y1, b1_x2, b1_y2;
	torch::Tensor b2_x1, b2_y1, b2_x2, b2_y2;
	if (xyxy) {
		b1_x1 = box1.select(-1, 0);
		b1_y1 = box1.select(-1, 1);
		b1_x2 = box1.select(-1, 2);
		b1_y2 = box1.select(-1, 3);
		//
		b2_x1 = box2.select(-1, 0);
		b2_y1 = box2.select(-1, 1);
		b2_x2 = box2.select(-1, 2);
		b2_y2 = box2.select(-1, 3);
	} else {
		b1_x1 = box1.select(-1, 0) - box1.select(-1, 2) / 2;
		b1_y1 = box1.select(-1, 1) - box1.select(-1, 3) / 2;
		b1_x2 = box1.select(-1, 0) + box1.select(-1, 2) / 2;
		b1_y2 = box1.select(-1, 1) + box1.select(-1, 3) / 2;
		//
		b2_x1 = box2.select(-1, 0) - box2.select(-1, 2) / 2;
		b2_y1 = box2.select(-1, 1) - box2.select(-1, 3) / 2;
		b2_x2 = box2.select(-1, 0) + box2.select(-1, 2) / 2;
		b2_y2 = box2.select(-1, 1) + box2.select(-1, 3) / 2;
	}

	// et the corrdinates of the intersection rectangle
	torch::Tensor inter_rect_x1 = torch::max(b1_x1, b2_x1);
	torch::Tensor inter_rect_y1 = torch::max(b1_y1, b2_y1);
	torch::Tensor inter_rect_x2 = torch::min(b1_x2, b2_x2);
	torch::Tensor inter_rect_y2 = torch::min(b1_y2, b2_y2);

	// Intersection area
	torch::Tensor inter_area = torch::max(inter_rect_x2 - inter_rect_x1 + 1,
			torch::zeros(inter_rect_x2.sizes()).to(device))
			* torch::max(inter_rect_y2 - inter_rect_y1 + 1,
					torch::zeros(inter_rect_x2.sizes()).to(device));

	// Union Area
	torch::Tensor b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1);
	torch::Tensor b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1);

	torch::Tensor iou = inter_area / (b1_area + b2_area - inter_area);

	return iou;
}

// outputs[batches_pred_num, 8] batch_index, left x, left y, right x, right y, object confidence, class_score, class_id
// targets[num_labels, 6] batch_index, class_id, left x, left y, right x, right y
torch::Tensor get_batch_statistics(torch::Tensor outputs, torch::Tensor targets,
		float iou_threshold) {
	if (outputs.dim() == 1)
		return torch::tensor( { 0, 0, -1}, torch::kFloat).to(outputs.device()).unsqueeze(0);
	std::vector < torch::Tensor > temp { };
	for (int sample_i = 0; sample_i < yolo_ops.num_classes; sample_i++) {
		torch::Tensor output = outputs.index(
				(outputs.select(1, 0) == sample_i).to(torch::kBool));
		if (!output.size(0))
			continue;
		torch::Tensor pred_boxes = output.slice(1, 1, 5);
		torch::Tensor pred_scores = output.select(1, 5);
		torch::Tensor pred_labels = output.select(1, -1);
		torch::Tensor true_positives = torch::zeros(pred_boxes.size(0)).to(
				targets.device());
		torch::Tensor annotations = targets.index(
				(targets.select(1, 0) == sample_i).to((torch::kBool))).slice(1,
				1, 6);
		if (annotations.size(0)) {
			torch::Tensor target_labels = annotations.select(1, 0);
			torch::Tensor target_boxes = annotations.slice(1, 1, 5);
			std::vector<int> detected_boxes { };
			for (int pred_i = 0; pred_i < pred_boxes.size(0); pred_i++) {
				torch::Tensor pred_box = pred_boxes[pred_i];
				torch::Tensor pred_label = pred_labels[pred_i];
				// If targets are found break
				if (detected_boxes.size() == annotations.size(0))
					break;
				// Ignore if label is not one of the target labels
				if (!(target_labels == pred_label).sum().item<int>())
					continue;
				torch::Tensor iou, box_index;
				std::tie(iou, box_index) = bbox_iou(pred_box.unsqueeze(0),
						target_boxes).max(0);
				std::vector<int>::iterator it = std::find(
						detected_boxes.begin(), detected_boxes.end(),
						box_index.item<int>());
				if (iou.item<float>() > iou_threshold
						&& it == detected_boxes.end()) {
					true_positives[pred_i] = 1;
					detected_boxes.push_back(box_index.item<int>());
				}
			}
		}
		temp.push_back(
				torch::cat( { true_positives.unsqueeze(1),
						pred_scores.unsqueeze(1), pred_labels.unsqueeze(1) },
						1));
	}
	if(temp.empty()){
		return torch::tensor( { 0, 0, -1}, torch::kFloat).to(outputs.device()).unsqueeze(0);
	}
	torch::Tensor batch_metrics = torch::cat(temp).to(targets.device());

	return batch_metrics;
}

torch::Tensor compute_ap(torch::Tensor r, torch::Tensor p) {
	// correct AP calculation
	// first append sentinel values at the end
	torch::Tensor mrec = torch::cat( { torch::tensor( { 0.0 }), r.to(torch::kCPU),
			torch::tensor( { 1.0 }) }, 0);
	torch::Tensor mpre = torch::cat( { torch::tensor( { 0.0 }), p.to(torch::kCPU),
			torch::tensor( { 0.0 }) }, 0);
	// compute the precision envelope
	for (int i = mpre.size(0) - 1; i > 0; i--) {
		mpre[i - 1] = torch::max(mpre[i - 1], mpre[i]);
	}
	// to calculate area under PR curve, look for points
    // where X axis (recall) changes value
	torch::Tensor idx = torch::where(mrec.slice(0, 1, mrec.size(0))
			!= mrec.slice(0, 0, -1))[0];
	torch::Tensor ap = torch::sum((mrec.index(idx + 1) - mrec.index(idx)) * mpre.index(idx + 1));

	return ap;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
		torch::Tensor> ap_per_class(torch::Tensor tp,
		torch::Tensor conf, torch::Tensor pred_cls,
		torch::Tensor target_cls) {
	// Sort by objectness
	torch::Tensor order = torch::argsort(-conf);
	tp = tp.index(order);
	conf = conf.index(order);
	pred_cls = pred_cls.index(order);
	// Find unique classes
	torch::Tensor unique_classes, tmp, tmpp;
	std::tie(unique_classes, tmp, tmpp) = torch::unique_dim(target_cls, 0);
	// Create Precision-Recall curve and compute AP for each class
	std::vector<torch::Tensor> ap, p, r;
	for(int i = 0; i < unique_classes.size(0); i++){
		torch::Tensor c = unique_classes[i];
		torch::Tensor idx = pred_cls == c;

		int n_gt = (target_cls == c).sum().item<int>();	// Number of ground truth objects
		int n_p = idx.sum().item<int>();	// Number of predicted objects

		if(!n_gt && !n_p)
			continue;
		else if(!n_gt || !n_p){
			torch::Tensor zero = torch::tensor({0}, torch::kFloat).squeeze(0).to(tp.device());
			ap.push_back(zero);
			p.push_back(zero);
			r.push_back(zero);
		}
		else{
			// Accumulate FPs and TPs
			torch::Tensor fpc = (1 - tp.index(idx)).cumsum(0);
			torch::Tensor tpc = (tp.index(idx)).cumsum(0);

			// Recall
			torch::Tensor recall_curve = tpc / (n_gt + 1e-16);
			r.push_back(recall_curve.select(0, -1));
			// Precision
			torch::Tensor precision_curve = tpc / (tpc + fpc);
			p.push_back(precision_curve.select(0, -1));
			// AP from recall-precision curve
			ap.push_back(compute_ap(recall_curve, precision_curve).to(tp.device()));
		}
	}
	// Compute F1 score (harmonic mean of precision and recall)
	torch::Tensor ap_t = torch::stack(ap);
	torch::Tensor p_t = torch::stack(p);
	torch::Tensor r_t = torch::stack(r);
	torch::Tensor f1 = 2 * p_t * r_t / (p_t + r_t + 1e-16);

	return {p_t, r_t, ap_t, f1, unique_classes};
}

#endif
