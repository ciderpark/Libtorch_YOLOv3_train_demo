#ifndef YOLOV3_HPP_
#define YOLOV3_HPP_
#include <stdio.h>
#include <iostream>
#include <typeinfo>
#include "yolov3.h"

// trim from start (in place)
static inline void
ltrim (std::string &s)
{
  s.erase (s.begin (), std::find_if (s.begin (), s.end (), [](int ch)
  {
    return !std::isspace(ch);
  }));
}

// trim from end (in place)
static inline void
rtrim (std::string &s)
{
  s.erase (std::find_if (s.rbegin (), s.rend (), [](int ch)
  {
    return !std::isspace(ch);
  }).base (),
           s.end ());
}

// trim from both ends (in place)
static inline void
trim (std::string &s)
{
  ltrim (s);
  rtrim (s);
}

static inline int
split (const string& str,
       std::vector<string>& ret_,
       string sep = ",")
{
  if (str.empty ())
  {
    return 0;
  }

  string tmp;
  string::size_type pos_begin = str.find_first_not_of (sep);
  string::size_type comma_pos = 0;

  while (pos_begin != string::npos)
  {
    comma_pos = str.find (sep, pos_begin);
    if (comma_pos != string::npos)
    {
      tmp = str.substr (pos_begin, comma_pos - pos_begin);
      pos_begin = comma_pos + sep.length ();
    }
    else
    {
      tmp = str.substr (pos_begin);
      pos_begin = comma_pos;
    }

    if (!tmp.empty ())
    {
      trim (tmp);
      ret_.push_back (tmp);
      tmp.clear ();
    }
  }
  return 0;
}

static inline int
split (const string& str,
       std::vector<int>& ret_,
       string sep = ",")
{
  std::vector < string > tmp;
  split (str, tmp, sep);

  for (size_t i = 0; i < tmp.size (); i++)
  {
    ret_.push_back (std::stoi (tmp[i]));
  }
  return 0;
}

// returns the IoU of two bounding boxes
static inline torch::Tensor
get_bbox_iou (torch::Tensor box1,
              torch::Tensor box2)
{
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

int
Darknet::get_int_from_cfg (map<string, string> block,
                           string key,
                           int default_value)
{
  if (block.find (key) != block.end ())
  {
    return std::stoi (block.at (key));
  }
  return default_value;
}

string
Darknet::get_string_from_cfg (map<string, string> block,
                              string key,
                              string default_value)
{
  if (block.find (key) != block.end ())
  {
    return block.at (key);
  }
  return default_value;
}

torch::nn::Conv2dOptions
conv_options (int64_t in_planes,
              int64_t out_planes,
              int64_t kerner_size,
              int64_t stride,
              int64_t padding,
              int64_t groups,
              bool with_bias = false)
{
  torch::nn::Conv2dOptions conv_options = torch::nn::Conv2dOptions (in_planes, out_planes, kerner_size);
  conv_options.stride (stride);
  conv_options.padding (padding);
  conv_options.groups (groups);
  conv_options.bias (with_bias);
  return conv_options;
}

torch::nn::BatchNorm2dOptions
bn_options (int64_t features)
{
  torch::nn::BatchNorm2dOptions bn_options = torch::nn::BatchNorm2dOptions (features);
  bn_options.affine (true);
  bn_options.track_running_stats (true);
  return bn_options;
}

struct EmptyLayer : torch::nn::Module
{
    EmptyLayer ()
    {

    }

    torch::Tensor
    forward (torch::Tensor x)
    {
      return x;
    }
};

struct UpsampleLayer : torch::nn::Module
{
    int _stride;
    UpsampleLayer (int stride)
    {
      _stride = stride;
    }

    torch::Tensor
    forward (torch::Tensor x)
    {

      torch::IntArrayRef sizes = x.sizes ();

      int64_t w, h;

      if (sizes.size () == 4)
      {
        w = sizes[2] * _stride;
        h = sizes[3] * _stride;

        x = torch::upsample_nearest2d (x, { w, h });
      }
      else if (sizes.size () == 3)
      {
        w = sizes[2] * _stride;
        x = torch::upsample_nearest1d (x, { w });
      }
      return x;
    }
};

struct MaxPoolLayer2D : torch::nn::Module
{
    int _kernel_size;
    int _stride;
    MaxPoolLayer2D (int kernel_size,
                    int stride)
    {
      _kernel_size = kernel_size;
      _stride = stride;
    }

    torch::Tensor
    forward (torch::Tensor x)
    {
      if (_stride != 1)
      {
        x = torch::max_pool2d (x, { _kernel_size, _kernel_size }, { _stride, _stride });
      }
      else
      {
        int pad = _kernel_size - 1;

        torch::Tensor padded_x = torch::replication_pad2d (x, { 0, pad, 0, pad });
        x = torch::max_pool2d (padded_x, { _kernel_size, _kernel_size }, { _stride, _stride });
      }

      return x;
    }
};

struct DetectionLayer : torch::nn::Module
{
    vector<float> _anchors;
    float ignore_thres;

    DetectionLayer (vector<float> anchors)
    {
      _anchors = anchors;
      ignore_thres = 0.5;
    }

    torch::Tensor
    forward (torch::Tensor prediction,
             int inp_dim,
             int num_classes,
             torch::Device device,
             torch::Tensor targets/* = torch::tensor({0})*/,
             torch::Tensor* layer_loss/* = nullptr*/)
    {
      return predict_transform (prediction, targets, layer_loss, inp_dim, _anchors, num_classes, device);
    }

    torch::Tensor
    predict_transform (torch::Tensor prediction,
                       torch::Tensor targets,
                       torch::Tensor* layer_loss,
                       int inp_dim,
                       vector<float> anchors,
                       int num_classes,
                       torch::Device device)
    {
      int batch_size = prediction.size (0);
      int stride = floor (inp_dim / prediction.size (2));
      int grid_size = floor (inp_dim / stride);
      int bbox_attrs = 5 + num_classes;
      int num_anchors = anchors.size () / 2;

      for (size_t i = 0; i < anchors.size (); i++)
      {
        anchors[i] = anchors[i] / stride;
      }

      torch::Tensor temp =
          prediction.view ( { batch_size, num_anchors, bbox_attrs, grid_size, grid_size }).permute ( { 0, 1, 3, 4, 2 }).contiguous ();
      torch::Tensor x = torch::sigmoid(temp.select (4, 0));
      torch::Tensor y = torch::sigmoid(temp.select (4, 1));
      torch::Tensor w = temp.select (4, 2);
      torch::Tensor h = temp.select (4, 3);
      torch::Tensor pred_conf = torch::sigmoid(temp.select (4, 4));
      torch::Tensor pred_cls = torch::sigmoid(temp.slice (4, 5, bbox_attrs));
      torch::Tensor pred_boxes = torch::zeros (temp.slice (4, 0, 4).sizes (), torch::kFloat).to(device);

      //
      torch::Tensor grid_x = torch::arange(grid_size).repeat({grid_size, 1}).view({1, 1, grid_size, grid_size}).toType(torch::kFloat).to (device);
      torch::Tensor grid_y = torch::arange(grid_size).repeat({grid_size, 1}).t().view({1, 1, grid_size, grid_size}).toType(torch::kFloat).to (device);

      pred_boxes.select (4, 0) = x.data () + grid_x;
      pred_boxes.select (4, 1) = y.data () + grid_y;
      //

      torch::Tensor scaled_anchors = torch::from_blob (anchors.data (), { num_anchors, 2 });
      //if (device != nullptr)
      scaled_anchors = scaled_anchors.to (device);
      //
      torch::Tensor anchor_w = scaled_anchors.slice (1, 0, 1).view ( { 1, num_anchors, 1, 1 });
      torch::Tensor anchor_h = scaled_anchors.slice (1, 1, 2).view ( { 1, num_anchors, 1, 1 });

      pred_boxes.select (4, 2) = torch::exp (w.data ()) * anchor_w;
      pred_boxes.select (4, 3) = torch::exp (h.data ()) * anchor_h;

      // output
      torch::Tensor output = torch::cat ( { pred_boxes.view ( { batch_size, -1, 4 }) * stride, pred_conf.view ( { batch_size, -1, 1 }), pred_cls.view ( {
          batch_size, -1, num_classes }) },
                                         -1);

      // loss
      if(NULL != layer_loss){
          torch::Tensor iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf;

          std::tie (iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf) = buildTargets (targets, pred_boxes, pred_cls, scaled_anchors,
                                                                                                               ignore_thres);

          obj_mask = obj_mask.to (torch::kBool);
          noobj_mask = noobj_mask.to (torch::kBool);

          torch::Tensor loss_x, loss_y, loss_w, loss_h, loss_conf_obj, loss_conf_noobj, loss_conf, loss_cls;

          torch::nn::MSELoss mse_loss;
          loss_x = mse_loss (x.index (obj_mask), tx.index (obj_mask));
          loss_y = mse_loss (y.index (obj_mask), ty.index (obj_mask));
          loss_w = mse_loss (w.index (obj_mask), tw.index (obj_mask));
          loss_h = mse_loss (h.index (obj_mask), th.index (obj_mask));

          torch::nn::BCELoss bce_loss;
          loss_conf_obj = bce_loss (pred_conf.index (obj_mask), tconf.index (obj_mask));
          loss_conf_noobj = bce_loss (pred_conf.index (noobj_mask), tconf.index (noobj_mask));
          int obj_scale = 1, noobj_scale = 100;
          loss_conf = obj_scale * loss_conf_obj + noobj_scale * loss_conf_noobj;

          loss_cls = bce_loss (pred_cls.index (obj_mask), tcls.index (obj_mask));

          torch::Tensor total_loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls;
          *layer_loss = total_loss;

          //
          torch::Tensor cls_acc, conf_obj, conf_noobj, conf50, iou50, iou75, detected_mask, precision, recall50, recall75;

          cls_acc = 100 * class_mask.index (obj_mask).mean();
          conf_obj = pred_conf.index (obj_mask).mean();
          conf_noobj = pred_conf.index (noobj_mask).mean();
          conf50 = (pred_conf > 0.5).to(torch::kFloat);
          iou50 = (iou_scores > 0.5).to(torch::kFloat);
          iou75 = (iou_scores > 0.75).to(torch::kFloat);
          detected_mask = conf50 * class_mask * tconf;
          precision = torch::sum(iou50 * detected_mask) / (conf50.sum() + 1e-16);
          recall50 = torch::sum(iou50 * detected_mask) / (obj_mask.sum() + 1e-16);
          recall75 = torch::sum(iou75 * detected_mask) / (obj_mask.sum() + 1e-16);

          metrics["loss"] = total_loss.item<float>();
          metrics["x"] = loss_x.item<float>();
          metrics["y"] = loss_y.item<float>();
          metrics["w"] = loss_w.item<float>();
          metrics["h"] = loss_h.item<float>();
          metrics["conf"] = loss_conf.item<float>();
          metrics["cls"] = loss_cls.item<float>();
          metrics["cls_acc"] = cls_acc.item<float>();
          metrics["recall50"] = recall50.item<float>();
          metrics["recall75"] = recall75.item<float>();
          metrics["precision"] = precision.item<float>();
          metrics["conf_obj"] = conf_obj.item<float>();
          metrics["conf_noobj"] = conf_noobj.item<float>();
          metrics["grid_size"] = float(grid_size);

          metrics_v.push_back(metrics);
      }

      return output;
    }
};

//---------------------------------------------------------------------------
// Darknet
//---------------------------------------------------------------------------
Darknet::Darknet (const string cfg_file,
                  torch::Device *device)
{

  load_cfg (cfg_file);

  _device = device;

  create_modules ();
}

void
Darknet::load_cfg (const string cfg_file)
{
  ifstream fs (cfg_file);
  string line;

  if (!fs)
  {
    std::cout << "Fail to load cfg file:" << cfg_file << endl;
    return;
  }

  while (getline (fs, line))
  {
    trim (line);

    if (line.empty ())
    {
      continue;
    }

    if (line.substr (0, 1) == "[")
    {
      map < string, string > block;

      string key = line.substr (1, line.length () - 2);
      block["type"] = key;

      blocks.push_back (block);
    }
    else
    {
      map < string, string > *block = &blocks[blocks.size () - 1];

      vector < string > op_info;

      split (line, op_info, "=");

      if (op_info.size () == 2)
      {
        string p_key = op_info[0];
        string p_value = op_info[1];
        block->operator[] (p_key) = p_value;
      }
    }
  }
  fs.close ();
}

void
Darknet::create_modules ()
{
  int prev_filters = 3;

  std::vector<int> output_filters;

  int index = 0;

  int filters = 0;

  for (int i = 0, len = blocks.size (); i < len; i++)
  {
    map < string, string > block = blocks[i];

    string layer_type = block["type"];

    // std::cout << index << "--" << layer_type << endl;

    torch::nn::Sequential module;

    if (layer_type == "net")
      continue;
    if (layer_type == "convolutional")
    {
      string activation = get_string_from_cfg (block, "activation", "");
      int batch_normalize = get_int_from_cfg (block, "batch_normalize", 0);
      filters = get_int_from_cfg (block, "filters", 0);
      int padding = get_int_from_cfg (block, "pad", 0);
      int kernel_size = get_int_from_cfg (block, "size", 0);
      int stride = get_int_from_cfg (block, "stride", 1);

      int pad = padding > 0 ? (kernel_size - 1) / 2 : 0;
      bool with_bias = batch_normalize > 0 ? false : true;

      torch::nn::Conv2d conv = torch::nn::Conv2d (conv_options (prev_filters, filters, kernel_size, stride, pad, 1, with_bias));
      module->push_back (conv);

      if (batch_normalize > 0)
      {
        torch::nn::BatchNorm2d bn = torch::nn::BatchNorm2d (bn_options (filters));
        module->push_back (bn);
      }

      if (activation == "leaky")
      {
        module->push_back (torch::nn::Functional (torch::leaky_relu, /*slope=*/0.1));
      }
    }
    else if (layer_type == "upsample")
    {
      int stride = get_int_from_cfg (block, "stride", 1);

      UpsampleLayer uplayer (stride);
      module->push_back (uplayer);
    }
    else if (layer_type == "maxpool")
    {
      int stride = get_int_from_cfg (block, "stride", 1);
      int size = get_int_from_cfg (block, "size", 1);

      MaxPoolLayer2D poolLayer (size, stride);
      module->push_back (poolLayer);
    }
    else if (layer_type == "shortcut")
    {
      // skip connection
      int from = get_int_from_cfg (block, "from", 0);
      block["from"] = std::to_string (from);

      blocks[i] = block;

      // placeholder
      EmptyLayer layer;
      module->push_back (layer);
    }
    else if (layer_type == "route")
    {
      // L 85: -1, 61
      string layers_info = get_string_from_cfg (block, "layers", "");

      std::vector < string > layers;
      split (layers_info, layers, ",");

      std::string::size_type sz;
      signed int start = std::stoi (layers[0], &sz);
      signed int end = 0;

      if (layers.size () > 1)
      {
        end = std::stoi (layers[1], &sz);
      }

      if (start > 0)
        start = start - index;

      if (end > 0)
        end = end - index;

      block["start"] = std::to_string (start);
      block["end"] = std::to_string (end);

      blocks[i] = block;

      // placeholder
      EmptyLayer layer;
      module->push_back (layer);

      if (end < 0)
      {
        filters = output_filters[index + start] + output_filters[index + end];
      }
      else
      {
        filters = output_filters[index + start];
      }
    }
    else if (layer_type == "yolo")
    {
      string mask_info = get_string_from_cfg (block, "mask", "");
      std::vector<int> masks;
      split (mask_info, masks, ",");

      string anchor_info = get_string_from_cfg (block, "anchors", "");
      std::vector<int> anchors;
      split (anchor_info, anchors, ",");

      std::vector<float> anchor_points;
      int pos;
      for (size_t i = 0; i < masks.size (); i++)
      {
        pos = masks[i];
        anchor_points.push_back (anchors[pos * 2]);
        anchor_points.push_back (anchors[pos * 2 + 1]);
      }

      DetectionLayer layer (anchor_points);
      module->push_back (layer);
    }
    else
    {
      cout << "unsupported operator:" << layer_type << endl;
    }

    prev_filters = filters;
    output_filters.push_back (filters);
    module_list.push_back (module);

    char *module_key = new char[strlen ("layer_") + sizeof (index) + 1];

    sprintf (module_key, "%s%d", "layer_", index);

    register_module (module_key, module);

    index += 1;
  }
}

map<string, string>*
Darknet::get_net_info ()
{
  if (blocks.size () > 0)
  {
    return &blocks[0];
  }
  return
  {};
}

void
Darknet::load_weights (const string weight_file)
{
  ifstream fs (weight_file, ios::binary);
  assert(fs.is_open());
  // header info: 5 * int32_t
  int32_t header_size = sizeof(int32_t) * 5;

  int64_t index_weight = 0;

  fs.seekg (0, fs.end);
  int64_t length = fs.tellg ();
  // skip header
  length = length - header_size;

  fs.seekg (header_size, fs.beg);
  float *weights_src = (float *) malloc (length);
  fs.read (reinterpret_cast<char*> (weights_src), length);

  fs.close ();

  at::TensorOptions options = torch::TensorOptions ().dtype (torch::kFloat32)/*.is_variable (true)*/;
  at::Tensor weights = torch::from_blob (weights_src, { length / 4 });

  int pre_layers = -1;
  if(weight_file.substr(weight_file.size() - 3, 3) == ".74"){
	  pre_layers = 75;
  }

  for (int i = 0; i < module_list.size (); i++)
  {
    if (i == pre_layers)
      break;
    map < string, string > module_info = blocks[i + 1];

    string module_type = module_info["type"];

    // only conv layer need to load weight
    if (module_type != "convolutional")
      continue;

    torch::nn::Sequential seq_module = module_list[i];

    auto conv_module = seq_module.ptr ()->ptr (0);
    torch::nn::Conv2dImpl *conv_imp = dynamic_cast<torch::nn::Conv2dImpl *> (conv_module.get ());

    int batch_normalize = get_int_from_cfg (module_info, "batch_normalize", 0);

    if (batch_normalize > 0)
    {
      // second module
      auto bn_module = seq_module.ptr ()->ptr (1);

      torch::nn::BatchNorm2dImpl *bn_imp = dynamic_cast<torch::nn::BatchNorm2dImpl *> (bn_module.get ());

      int num_bn_biases = bn_imp->bias.numel ();

      at::Tensor bn_bias = weights.slice (0, index_weight, index_weight + num_bn_biases);
      index_weight = index_weight + num_bn_biases;

      at::Tensor bn_weights = weights.slice (0, index_weight, index_weight + num_bn_biases);
      index_weight = index_weight + num_bn_biases;

      at::Tensor bn_running_mean = weights.slice (0, index_weight, index_weight + num_bn_biases);
      index_weight = index_weight + num_bn_biases;

      at::Tensor bn_running_var = weights.slice (0, index_weight, index_weight + num_bn_biases);
      index_weight = index_weight + num_bn_biases;

      bn_bias = bn_bias.view_as (bn_imp->bias);
      bn_weights = bn_weights.view_as (bn_imp->weight);
      bn_running_mean = bn_running_mean.view_as (bn_imp->running_mean);
      bn_running_var = bn_running_var.view_as (bn_imp->running_var);

      bn_imp->bias.set_data (bn_bias);
      bn_imp->weight.set_data (bn_weights);
      bn_imp->running_mean.set_data (bn_running_mean);
      bn_imp->running_var.set_data (bn_running_var);
    }
    else
    {
      int num_conv_biases = conv_imp->bias.numel ();

      at::Tensor conv_bias = weights.slice (0, index_weight, index_weight + num_conv_biases);
      index_weight = index_weight + num_conv_biases;

      conv_bias = conv_bias.view_as (conv_imp->bias);
      conv_imp->bias.set_data (conv_bias);
    }

    int num_weights = conv_imp->weight.numel ();

    at::Tensor conv_weights = weights.slice (0, index_weight, index_weight + num_weights);
    index_weight = index_weight + num_weights;

    conv_weights = conv_weights.view_as (conv_imp->weight);
    conv_imp->weight.set_data (conv_weights);
  }
}

torch::Tensor
Darknet::forward (torch::Tensor x)
{
  int module_count = module_list.size ();

  std::vector < torch::Tensor > outputs (module_count);

  torch::Tensor result;
  int write = 0;

  for (int i = 0; i < module_count; i++)
  {
    map < string, string > block = blocks[i + 1];

    string layer_type = block["type"];

    if (layer_type == "net")
      continue;

    if (layer_type == "convolutional" || layer_type == "upsample" || layer_type == "maxpool")
    {
      torch::nn::SequentialImpl *seq_imp = dynamic_cast<torch::nn::SequentialImpl *> (module_list[i].ptr ().get ());

      x = seq_imp->forward (x);
      outputs[i] = x;
    }
    else if (layer_type == "route")
    {
      int start = std::stoi (block["start"]);
      int end = std::stoi (block["end"]);

      if (start > 0)
        start = start - i;

      if (end == 0)
      {
        x = outputs[i + start];
      }
      else
      {
        if (end > 0)
          end = end - i;

        torch::Tensor map_1 = outputs[i + start];
        torch::Tensor map_2 = outputs[i + end];

        x = torch::cat ( { map_1, map_2 }, 1);
      }

      outputs[i] = x;
    }
    else if (layer_type == "shortcut")
    {
      int from = std::stoi (block["from"]);
      x = outputs[i - 1] + outputs[i + from];
      outputs[i] = x;
    }
    else if (layer_type == "yolo")
    {
      torch::nn::SequentialImpl *seq_imp = dynamic_cast<torch::nn::SequentialImpl *> (module_list[i].ptr ().get ());

      map < string, string > net_info = blocks[0];
      int inp_dim = get_int_from_cfg (net_info, "height", 0);
      int num_classes = get_int_from_cfg (block, "classes", 0);
      torch::Tensor* null_t = NULL;
      x = seq_imp->forward (x, inp_dim, num_classes, *_device, torch::tensor({0}), null_t);
      delete null_t;
      if (write == 0)
      {
        result = x;
        write = 1;
      }
      else
      {
        result = torch::cat ( { result, x }, 1);
      }

      outputs[i] = outputs[i - 1];
    }
  }
  return result;
}

std::tuple<torch::Tensor, torch::Tensor>
Darknet::forward (torch::Tensor x,
                  torch::Tensor targets)
{
	metrics_v.clear();
  int module_count = module_list.size ();

  std::vector < torch::Tensor > outputs (module_count);

  torch::Tensor result;
  int write = 0;
  torch::Tensor loss;
  for (int i = 0; i < module_count; i++)
  {
    map < string, string > block = blocks[i + 1];

    string layer_type = block["type"];

    if (layer_type == "net")
      continue;

    if (layer_type == "convolutional" || layer_type == "upsample" || layer_type == "maxpool")
    {
      torch::nn::SequentialImpl *seq_imp = dynamic_cast<torch::nn::SequentialImpl *> (module_list[i].ptr ().get ());

      x = seq_imp->forward (x);
      outputs[i] = x;
    }
    else if (layer_type == "route")
    {
      int start = std::stoi (block["start"]);
      int end = std::stoi (block["end"]);

      if (start > 0)
        start = start - i;

      if (end == 0)
      {
        x = outputs[i + start];
      }
      else
      {
        if (end > 0)
          end = end - i;

        torch::Tensor map_1 = outputs[i + start];
        torch::Tensor map_2 = outputs[i + end];

        x = torch::cat ( { map_1, map_2 }, 1);
      }

      outputs[i] = x;
    }
    else if (layer_type == "shortcut")
    {
      int from = std::stoi (block["from"]);
      x = outputs[i - 1] + outputs[i + from];
      outputs[i] = x;
    }
    else if (layer_type == "yolo")
    {
      torch::nn::SequentialImpl *seq_imp = dynamic_cast<torch::nn::SequentialImpl *> (module_list[i].ptr ().get ());

      map < string, string > net_info = blocks[0];
      int inp_dim = get_int_from_cfg (net_info, "height", 0);
      int num_classes = get_int_from_cfg (block, "classes", 0);
      torch::Tensor layer_loss;
      x = seq_imp->forward (x, inp_dim, num_classes, *_device, targets, &layer_loss);

      if (write == 0)
      {
        result = x;
        loss = layer_loss;
        write = 1;
      }
      else
      {
        result = torch::cat ( { result, x }, 1);
        loss += layer_loss;
      }

      outputs[i] = outputs[i - 1];
    }
  }

  return
  { result, loss};
}

// batch_index, left x, left y, right x, right y, object confidence, class_score, class_id
torch::Tensor
Darknet::write_results (torch::Tensor prediction,
                        int num_classes,
                        float confidence,
                        float nms_conf)
{
  // get result which object confidence > threshold
  auto conf_mask = (prediction.select (2, 4) > confidence).to (torch::kFloat32).unsqueeze (2);

  prediction.mul_ (conf_mask);
  auto ind_nz = torch::nonzero (prediction.select (2, 4)).transpose (0, 1).contiguous ();

  if (ind_nz.size (0) == 0)
  {
    return torch::zeros ( { 0 });
  }

  torch::Tensor box_a = torch::ones (prediction.sizes (), prediction.options ());
  // top left x = centerX - w/2
  box_a.select (2, 0) = prediction.select (2, 0) - prediction.select (2, 2).div (2);
  box_a.select (2, 1) = prediction.select (2, 1) - prediction.select (2, 3).div (2);
  box_a.select (2, 2) = prediction.select (2, 0) + prediction.select (2, 2).div (2);
  box_a.select (2, 3) = prediction.select (2, 1) + prediction.select (2, 3).div (2);

  prediction.slice (2, 0, 4) = box_a.slice (2, 0, 4);

  int batch_size = prediction.size (0);
  int item_attr_size = 5;

  torch::Tensor output = torch::ones ( { 1, prediction.size (2) + 1 });
  bool write = false;

  int num = 0;

  for (int i = 0; i < batch_size; i++)
  {
    auto image_prediction = prediction[i];

    // get the max classes score at each result
    std::tuple < torch::Tensor, torch::Tensor > max_classes = torch::max (image_prediction.slice (1, item_attr_size, item_attr_size + num_classes), 1);

    // class score
    auto max_conf = std::get < 0 > (max_classes);

    // index
    auto max_conf_score = std::get < 1 > (max_classes);
    max_conf = max_conf.to (torch::kFloat32).unsqueeze (1);
    max_conf_score = max_conf_score.to (torch::kFloat32).unsqueeze (1);

    // shape: n * 7, left x, left y, right x, right y, object confidence, class_score, class_id
    image_prediction = torch::cat ( { image_prediction.slice (1, 0, 5), max_conf, max_conf_score }, 1);

    // remove item which object confidence == 0
    auto non_zero_index = torch::nonzero (image_prediction.select (1, 4));

    auto image_prediction_data = image_prediction.index_select (0, non_zero_index.squeeze ()).view ( { -1, 7 });

    // get unique classes z
    std::vector < torch::Tensor > img_classes;

    for (int m = 0, len = image_prediction_data.size (0); m < len; m++)
    {
      bool found = false;
      for (size_t n = 0; n < img_classes.size (); n++)
      {
        auto ret = (image_prediction_data[m][6] == img_classes[n]);
        if (torch::nonzero (ret).size (0) > 0)
        {
          found = true;
          break;
        }
      }
      if (!found)
        img_classes.push_back (image_prediction_data[m][6]);
    }

    for (size_t k = 0; k < img_classes.size (); k++)
    {
      auto cls = img_classes[k];

      auto cls_mask = image_prediction_data * (image_prediction_data.select (1, 6) == cls).to (torch::kFloat32).unsqueeze (1);
      auto class_mask_index = torch::nonzero (cls_mask.select (1, 5)).squeeze ();

      auto image_pred_class = image_prediction_data.index_select (0, class_mask_index).view ( { -1, 7 });
      // ascend by confidence
      // seems that inverse method not work
      std::tuple < torch::Tensor, torch::Tensor > sort_ret = torch::sort (image_pred_class.select (1, 4));

      auto conf_sort_index = std::get < 1 > (sort_ret);

      // seems that there is something wrong with inverse method
      // conf_sort_index = conf_sort_index.inverse();

      image_pred_class = image_pred_class.index_select (0, conf_sort_index.squeeze ()).cpu ();

      for (int w = 0; w < image_pred_class.size (0) - 1; w++)
      {
        int mi = image_pred_class.size (0) - 1 - w;

        if (mi <= 0)
        {
          break;
        }

        auto ious = get_bbox_iou (image_pred_class[mi].unsqueeze (0), image_pred_class.slice (0, 0, mi));

        auto iou_mask = (ious < nms_conf).to (torch::kFloat32).unsqueeze (1);
        image_pred_class.slice (0, 0, mi) = image_pred_class.slice (0, 0, mi) * iou_mask;

        // remove from list
        auto non_zero_index = torch::nonzero (image_pred_class.select (1, 4)).squeeze ();
        image_pred_class = image_pred_class.index_select (0, non_zero_index).view ( { -1, 7 });
      }

      torch::Tensor batch_index = torch::ones ( { image_pred_class.size (0), 1 }).fill_ (i);

      if (!write)
      {
        output = torch::cat ( { batch_index, image_pred_class }, 1);
        write = true;
      }
      else
      {
        auto out = torch::cat ( { batch_index, image_pred_class }, 1);
        output = torch::cat ( { output, out }, 0);
      }

      num += 1;
    }
  }

  if (num == 0)
  {
    return torch::zeros ( { 0 });
  }

  return output;
}
#endif
