#define STB_IMAGE_IMPLEMENTATION
#include "./stb/stb_image.h"
#include "./stb/drawing.hpp"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "./stb/stb_image_write.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "./stb/stb_image_resize.h"
static const uint8_t color[3] = {0xff, 0, 0};

#include "inference_nv12.h"
#include "venus.h"
#include <math.h>
#include <fstream>
#include <iostream>
#include <memory.h>
#include <memory>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <limits>
#include <cstring>
#include "debug.h"
#define IS_ALIGN_64(x) (((size_t)x) & 0x3F)


using namespace std;
using namespace magik::venus;

extern std::unique_ptr<venus::BaseNet> face_net;
extern std::unique_ptr<venus::BaseNet> emo_net;

typedef struct
{
    unsigned char* image;  
    int w;
    int h;
}input_info_t;


struct PixelOffset {
    int top;
    int bottom;
    int left;
    int right;
};

void check_pixel_offset(PixelOffset &pixel_offset){
    // 5 5 -> 6 4
    // padding size not is Odd number
    if(pixel_offset.top % 2 == 1){
        pixel_offset.top += 1;
        pixel_offset.bottom -=1;
    }
    if(pixel_offset.left % 2 == 1){
        pixel_offset.left += 1;
        pixel_offset.right -=1;
    }
}

uint8_t* read_bin(const char* path)
{
    std::ifstream infile;
    infile.open(path, std::ios::binary | std::ios::in);
    infile.seekg(0, std::ios::end);
    int length = infile.tellg();
    infile.seekg(0, std::ios::beg);
    uint8_t* buffer_pointer = new uint8_t[length];
    infile.read((char*)buffer_pointer, length);
    infile.close();
    return buffer_pointer;
}

std::vector<std::string> splitString(std::string srcStr, std::string delimStr,bool repeatedCharIgnored = false)
{
    std::vector<std::string> resultStringVector;
    std::replace_if(srcStr.begin(), srcStr.end(), [&](const char& c){if(delimStr.find(c)!=std::string::npos){return true;}else{return false;}}, delimStr.at(0));
    size_t pos=srcStr.find(delimStr.at(0));
    std::string addedString="";
    while (pos!=std::string::npos) {
        addedString=srcStr.substr(0,pos);
        if (!addedString.empty()||!repeatedCharIgnored) {
            resultStringVector.push_back(addedString);
        }
        srcStr.erase(srcStr.begin(), srcStr.begin()+pos+1);
        pos=srcStr.find(delimStr.at(0));
    }
    addedString=srcStr;
    if (!addedString.empty()||!repeatedCharIgnored) {
        resultStringVector.push_back(addedString);
    }
    return resultStringVector;
}

void write_input_bin(std::unique_ptr<const venus::Tensor>& tensor, std::string name = "mnn.bin") {
	std::ofstream outFile(name, std::ios::out | std::ios::binary);
	auto shape = tensor->shape();
    std::cout << "input shape: " << std::endl;
	int size = 1;
	for(auto s : shape) {
        std::cout << s << ",";
		size *= s;
	}
    std::cout << std::endl;
	const uint8_t *data = tensor->data<uint8_t>();
	outFile.write((char*)data, sizeof(uint8_t)*size);
    outFile.close();
}

void write_output_bin(std::unique_ptr<const venus::Tensor>& tensor, std::string name = "mnn.bin") {
	std::ofstream outFile(name, std::ios::out | std::ios::binary);
	auto shape = tensor->shape();
    std::cout << "output shape: " << std::endl;
	int size = 1;
	for(auto s : shape) {
        std::cout << s << ",";
		size *= s;
	}
    std::cout << std::endl;
	const float *data = tensor->data<float>();
	outFile.write((char*)data, sizeof(float)*size);
    outFile.close();

    // int len = std::min(100, size);
    // for(int i = 0; i < len; i++){
    //     printf("%f, ", data[i]);
    // }
    // printf("\n");
}

vector<vector<float>> min_boxes = {{10.0, 16.0, 24.0}, {32.0, 48.0}, {64.0, 96.0}, {128.0, 192.0, 256.0}};
vector<float> strides = {8.0, 16.0, 32.0, 64.0};

vector<vector<float>> generate_priors(const vector<vector<int>>& feature_map_list, const vector<vector<float>>& shrinkage_list, const vector<int>& image_size, const vector<vector<float>>& min_boxes) {
    vector<vector<float>> priors;
    for (size_t index = 0; index < feature_map_list[0].size(); ++index) {
        float scale_w = image_size[0] / shrinkage_list[0][index];
        float scale_h = image_size[1] / shrinkage_list[1][index];
        for (int j = 0; j < feature_map_list[1][index]; ++j) {
            for (int i = 0; i < feature_map_list[0][index]; ++i) {
                float x_center = (i + 0.5) / scale_w;
                float y_center = (j + 0.5) / scale_h;

                for (float min_box : min_boxes[index]) {
                    float w = min_box / image_size[0];
                    float h = min_box / image_size[1];
                    priors.push_back({x_center, y_center, w, h});
                }
            }
        }
    }
    // cout << "priors nums:" << priors.size() << endl;
    // Clipping the priors to be within [0.0, 1.0]
    // for (auto& prior : priors) {  
    //     for (auto& val : prior) {  
    //         val = std::min(std::max(val, 0.0f), 1.0f);  
    //     }  
    // } 
    for(int i = 0; i < priors.size(); i++){
		for(int j = 0; j < priors[i].size(); j++){
			float val = priors[i][j];
			float value_b =  std::min(std::max(val, 0.0f), 1.0f);
			priors[i][j] = value_b;
		}
	}  

    return priors;
}

vector<vector<float>> define_img_size(const vector<int>& image_size) {
    vector<vector<int>> feature_map_w_h_list;
    vector<vector<float>> shrinkage_list;
    for (int size : image_size) {
        vector<int> feature_map;
        for (float stride : strides) {
            feature_map.push_back(static_cast<int>(ceil(size / stride)));
        }
        feature_map_w_h_list.push_back(feature_map);
    }

    for (size_t i = 0; i < image_size.size(); ++i) {
        shrinkage_list.push_back(strides);
    }
    return generate_priors(feature_map_w_h_list, shrinkage_list, image_size, min_boxes);
}

vector<vector<float>> convert_locations_to_boxes(const vector<vector<float>>& locations, const vector<vector<float>>& priors, float center_variance, float size_variance) {
    vector<vector<float>> boxes;
    for (size_t i = 0; i < locations.size(); ++i) {
        vector<float> box;
        for (size_t j = 0; j < locations[i].size() / 4; ++j) {
            float cx = locations[i][j * 4 + 0] * center_variance * priors[i][2] + priors[i][0];
            float cy = locations[i][j * 4 + 1] * center_variance * priors[i][3] + priors[i][1];
            float w = exp(locations[i][j * 4 + 2] * size_variance) * priors[i][2];
            float h = exp(locations[i][j * 4 + 3] * size_variance) * priors[i][3];
            box.push_back(cx);
            box.push_back(cy);
            box.push_back(w);
            box.push_back(h);
        }
        boxes.push_back(box);
    }
    return boxes;
}

vector<vector<float>> center_form_to_corner_form(const vector<vector<float>>& locations) {
    vector<vector<float>> boxes;
    for (const auto& loc : locations) {
        vector<float> box;
        for (size_t i = 0; i < loc.size() / 4; ++i) {
            float cx = loc[i * 4 + 0];
            float cy = loc[i * 4 + 1];
            float w = loc[i * 4 + 2];
            float h = loc[i * 4 + 3];
            float xmin = cx - w / 2;
            float ymin = cy - h / 2;
            float xmax = cx + w / 2;
            float ymax = cy + h / 2;
            box.push_back(xmin);
            box.push_back(ymin);
            box.push_back(xmax);
            box.push_back(ymax);
        }
        boxes.push_back(box);
    }
    return boxes;
}

float area_of(float left, float top, float right, float bottom) {
    float width = max(0.0f, right - left);
    float height = max(0.0f, bottom - top);
    return width * height;
}

float iou_of(const vector<float>& box0, const vector<float>& box1) {
    float overlap_left = max(box0[0], box1[0]);
    float overlap_top = max(box0[1], box1[1]);
    float overlap_right = min(box0[2], box1[2]);
    float overlap_bottom = min(box0[3], box1[3]);

    float overlap_area = area_of(overlap_left, overlap_top, overlap_right, overlap_bottom);
    float area0 = area_of(box0[0], box0[1], box0[2], box0[3]);
    float area1 = area_of(box1[0], box1[1], box1[2], box1[3]);
    float total_area = area0 + area1 - overlap_area;
    if (total_area <= 0.0f) return 0.0f;
    return overlap_area / total_area;
}

vector<vector<float>> hard_nms(const vector<vector<float>>& box_scores, float iou_threshold, int top_k = -1, int candidate_size = 200) {
    vector<int> idx(box_scores.size());
    iota(idx.begin(), idx.end(), 0);

    sort(idx.begin(), idx.end(), [&box_scores](int i1, int i2) {
        return box_scores[i1].back() < box_scores[i2].back();
    });

    if (candidate_size > 0 && candidate_size < (int)idx.size()) {
        idx.resize(candidate_size);
    }

    vector<vector<float>> picked;
    while (!idx.empty()) {
        int current = idx.back();
        const auto& current_box = box_scores[current];
        picked.push_back(current_box);
        if (top_k > 0 && (int)picked.size() >= top_k) break;
        idx.pop_back();

        for (auto it = idx.begin(); it != idx.end();) {
            float iou = iou_of(box_scores[*it], current_box);
            if (iou > iou_threshold) {
                it = idx.erase(it);
            } else {
                ++it;
            }
        }
    }
    return picked;
}

vector<vector<float>> predict(float width, float height, const vector<vector<float>>& scores, const vector<vector<float>>& boxes, float prob_threshold, float iou_threshold = 0.3, int top_k = -1) {
    vector<vector<float>> final_boxes;
    vector<vector<float>> box_scores; // Combine boxes and scores in the required format
    for (size_t i = 0; i < boxes.size(); ++i) {
        vector<float> box_score = boxes[i];
        box_score.push_back(scores[i][1]); // Assuming class score is at index 1
        if (scores[i][1] > prob_threshold) {
            box_scores.push_back(box_score);
        }
    }
    
    vector<vector<float>> picked = hard_nms(box_scores, iou_threshold, top_k);

    // Convert coordinates back to original scale and print
    for (const auto& box : picked) {
        // cout << "Box: ";
        vector<float> face_box;
        for (size_t i = 0; i < 4; ++i) {
            float coord = i % 2 == 0 ? box[i] * width : box[i] * height;
            face_box.push_back((int)coord);
            // cout << coord << " ";
        }
        final_boxes.push_back(face_box);
        // cout << "Score: " << box.back() << endl;
    }
    return final_boxes;
}

void softmax(const float* input, float* output, int w, int h, int c) {
    const float* in_data = input;
    int first = h;
    int second = c;
    int third = w;

    int softmax_size = w * h;
    float* softmax_data = (float*)malloc(softmax_size * sizeof(float));
    float* max = (float*)malloc(softmax_size * sizeof(float));

    if (softmax_data == NULL || max == NULL) {  
        // Handle memory allocation failure  
        if (softmax_data) free(softmax_data);  
        if (max) free(max);  
        return;  

    }

    for (int f = 0; f < first; ++f) {
        for (int t = 0; t < third; ++t) {
            int m_under = f * third + t;
            max[m_under] = -FLT_MAX;
            for (int s = 0; s < second; ++s) {
                int i_under = f * third * second + s * third + t;
                max[m_under] = in_data[i_under] > max[m_under] ? in_data[i_under] : max[m_under];
            }
            softmax_data[m_under] = 0;
            for (int s = 0; s < second; ++s) {
                int i_under = f * third * second + s * third + t;
                float temp = in_data[i_under];
                softmax_data[m_under] += exp(temp - max[m_under]);
            }
            for (int s = 0; s < second; ++s) {
                int i_under = f * third * second + s * third + t;
                float input_num = in_data[i_under];
                float softmax_num = exp(input_num - max[m_under]) / softmax_data[m_under];
                output[i_under] = softmax_num;
            }
        }
    }
    // Free the allocated memory  
    free(softmax_data);  
    free(max);
}

int Goto_Magik_Detect(char * nv12Data, int width, int height){
    int ret = 0;
    /* set cpu affinity */
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(0, &mask);
    if (-1 == sched_setaffinity(0, sizeof(mask), &mask)) {
        fprintf(stderr, "set cpu affinity failed, %s\n", strerror(errno));
        return -1;
    }

    bool cvtbgra;
    cvtbgra = false;
    void *handle = NULL;

    int ori_img_h = -1;
    int ori_img_w = -1;
    float scale = 1.0;
    int face_in_w = 320, face_in_h = 240;
    int emo_in_w = 48, emo_in_h = 48;
    std::string emotion_array[] = {"neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear"};

    PixelOffset pixel_offset;
    std::unique_ptr<venus::Tensor> input;
    std::unique_ptr<venus::Tensor> emo_input;

    input_info_t input_src;
    input_src.w = width;
    input_src.h = height;
    input_src.image = (unsigned char*)nv12Data;

    //---------------------process-------------------------------
    // get ori image w h
    ori_img_w = input_src.w;
    ori_img_h = input_src.h;

    // printf("ori_image w,h: %d ,%d \n",ori_img_w,ori_img_h);
    // unsigned char *imagedata = stbi_load(argv[3], &ori_img_w, &ori_img_h, &comp, 3); // RGB
    
    input = face_net->get_input(0);
    magik::venus::shape_t input_shape = input->shape();
    // printf("face model-->%d %d %d \n",input_shape[1], input_shape[2], input_shape[3]);

    if (cvtbgra)
    {
        input->reshape({1, face_in_h, face_in_w , 4});
    }else
    {
        input->reshape({1, face_in_h, face_in_w, 1});
    }
  

    // uint8_t *indata = input->mudata<uint8_t>();
    
    //resize and padding
    magik::venus::Tensor temp_ori_input({1, ori_img_h, ori_img_w, 1}, TensorFormat::NV12);
    uint8_t *tensor_data = temp_ori_input.mudata<uint8_t>();
    int src_size = int(ori_img_h * ori_img_w * 1.5);
    magik::venus::memcopy((void*)tensor_data, (void*)input_src.image, src_size * sizeof(uint8_t));

    // float scale_x = (float)face_in_w/(float)ori_img_w;
    // float scale_y = (float)face_in_h/(float)ori_img_h;
    // scale = scale_x < scale_y ? scale_x:scale_y;  //min scale
    // printf("scale---> %f\n",scale);

    // int valid_dst_w = (int)(scale*ori_img_w);
    // if (valid_dst_w % 2 == 1)
    //     valid_dst_w = valid_dst_w + 1;
    // int valid_dst_h = (int)(scale*ori_img_h);
    // if (valid_dst_h % 2 == 1)
    //     valid_dst_h = valid_dst_h + 1;

    // int dw = face_in_w - valid_dst_w;
    // int dh = face_in_h - valid_dst_h;

    // pixel_offset.top = int(round(float(dh)/2 - 0.1));
    // pixel_offset.bottom = int(round(float(dh)/2 + 0.1));
    // pixel_offset.left = int(round(float(dw)/2 - 0.1));
    // pixel_offset.right = int(round(float(dw)/2 + 0.1));
    
//    check_pixel_offset(pixel_offset);

    magik::venus::BsCommonParam param;
    param.pad_val = 0;
    param.pad_type = magik::venus::BsPadType::NONE;
    param.input_height = ori_img_h;
    param.input_width = ori_img_w;
    param.input_line_stride = ori_img_w;
    param.in_layout = magik::venus::ChannelLayout::NV12;

    if (cvtbgra)
    {
        param.out_layout = magik::venus::ChannelLayout::RGBA;
    }else
    {
        param.out_layout = magik::venus::ChannelLayout::NV12;
    }
    magik::venus::common_resize((const void*)tensor_data, *input.get(), magik::venus::AddressLocate::NMEM_VIRTUAL, &param);
    face_net->run();

	// std::vector<std::string> output_names = face_net->get_output_names();
	// for (int i = 0; i < int(output_names.size()); i++) {
    //     std::cout << "output_" << i << ": " << output_names[i] << std::endl;
    //     std::unique_ptr<const venus::Tensor> output_tensor = face_net->get_output(i);
    //     std::string output_name = output_names[i] + ".bin";
    //     write_output_bin(output_tensor, output_name);
	// }

    // postprocessing
    std::unique_ptr<const venus::Tensor> out_0 = face_net->get_output(0); 
    std::unique_ptr<const venus::Tensor> out_1 = face_net->get_output(1);

    const float* output_data_0 = out_0->data<float>();
    const float* output_data_1 = out_1->data<float>();

    auto shape_0 = out_0->shape(); // scores
    auto shape_1 = out_1->shape(); // boxes

    int scores_size = shape_0[0]*shape_0[1]*shape_0[2]; // 1,4420,2
    int boxes_size  = shape_1[0]*shape_1[1]*shape_1[2]; // 1,4420,4,

    float* output_data_0_softmax = (float*)malloc(scores_size * sizeof(float));
    softmax(output_data_0, output_data_0_softmax, shape_0[0], shape_0[1], shape_0[2]);

    vector<vector<float>> scores;
    vector<vector<float>> boxes;

    // Assuming shape_0[1] == shape_1[1]: give the number of detections
    for (int i = 0; i < shape_0[1]; ++i) {
        // Extract scores
        vector<float> score;
        for (int j = 0; j < shape_0[2]; ++j) {
            score.push_back(output_data_0_softmax[i * shape_0[2] + j]);
        }
        scores.push_back(score);
  
        // Extract boxes
        vector<float> box;
        // Assuming shape_0[2] == 4, for [x1, y1, x2, y2]
        for (int k = 0; k < shape_1[2]; ++k) {
            box.push_back(output_data_1[i * shape_1[2] + k]);
        }
        boxes.push_back(box);
    }
    free(output_data_0_softmax);

    vector<int> input_size = {320, 240};
    float center_variance = 0.1;
    float size_variance = 0.2;
    float prob_threshold = 0.7;
    float iou_threshold = 0.3;
    int top_k = -1;

    vector<vector<float>> priors = define_img_size(input_size);
    
    vector<vector<float>> converted_boxes = convert_locations_to_boxes(boxes, priors, center_variance, size_variance);

    vector<vector<float>> final_boxes = center_form_to_corner_form(converted_boxes);

    vector<vector<float>> final_face_boxes = predict(ori_img_w, ori_img_h, scores, final_boxes, prob_threshold, iou_threshold, top_k);

    if (final_face_boxes.size() > 0){
        // cout << final_face_boxes.size() << " Face Detected!!! " << endl;
        emo_input = emo_net->get_input(0);
        magik::venus::shape_t emo_input_shape = emo_input->shape();
        // printf("emotion detection model-->%d %d %d \n",emo_input_shape[1], emo_input_shape[2], emo_input_shape[3]);
    }
    else{
        return 3;
    }

//######################################################################

    int n_face = 1;
    for (const auto& face_box : final_face_boxes) {  
        int roi_x = face_box[0];  
        int roi_y = face_box[1];  
        int roi_w = face_box[2] - face_box[0];  
        int roi_h = face_box[3] - face_box[1];  
    
        // Adjust coordinates to be within the image bounds  
        roi_x = std::max(roi_x, 0);  
        roi_y = std::max(roi_y, 0);  
    
        // Adjust width and height to be within the image bounds  
        roi_w = std::min(roi_w, ori_img_w - roi_x);  
        roi_h = std::min(roi_h, ori_img_h - roi_y);  
        if (roi_w % 2 == 1) roi_w += 1;  
        if (roi_h % 2 == 1) roi_h += 1;  
    
        // Check if the adjusted ROI is still valid  
        if (roi_w <= 0 || roi_h <= 0) {  
            std::cerr << "Invalid ROI dimensions after adjustment\n";  
            continue;  
        }  
    
        // std::cout << "face_" << n_face << ": " << roi_x << " " << roi_y << " " << roi_w << " " << roi_h << endl;  
    
        // Calculate the start of the Y plane  
        const unsigned char* yPlane = reinterpret_cast<const unsigned char*>(nv12Data);  
        // Calculate the start of the UV plane (not used in this function but important to know)  
        const unsigned char* uvPlane = yPlane + (ori_img_w * ori_img_h);  
    
     
        // Resize the output grayscale data vector to the size of the cropped region 
        unsigned char* grayscaleData = new unsigned char[roi_w * roi_h];  
    
        // Iterate over the ROI and copy the Y (luminance) values to the grayscale output  
        for (int y = 0; y < roi_h; ++y) {  
            int srcYIndex = (roi_y + y) * ori_img_w + roi_x;  
            int dstYIndex = y * roi_w;  
            std::memcpy(&grayscaleData[dstYIndex], &yPlane[srcYIndex], roi_w);  
        }  

        n_face++;  

        // Resize grayscale image to model's expected input dimensions
        unsigned char* resizedData = new unsigned char[emo_in_w * emo_in_h];
        stbir_resize_uint8(grayscaleData, roi_w, roi_h, 0, resizedData, emo_in_w, emo_in_h, 0, 1);

        uint8_t *indata = emo_input->mudata<uint8_t>();
       
        for (int i = 0; i < emo_in_h; i ++)
        {
            for (int j = 0; j < emo_in_w; j++)
            {
                int data1 = resizedData[i*emo_in_w + j];
                indata[i*emo_in_w*4 + j*4 + 0] = data1;
                indata[i*emo_in_w*4 + j*4 + 1] = 0;
                indata[i*emo_in_w*4 + j*4 + 2] = 0;
                indata[i*emo_in_w*4 + j*4 + 3] = 0;
            }
        }

        emo_net->run();

        // Process model output
        std::unique_ptr<const venus::Tensor> output_tensor = emo_net->get_output(0); // Assuming single output
        const float* output_data = output_tensor->data<float>();

        int num_classes = output_tensor->shape()[3]; // Assuming the output shape is [1, num_classes]


        // Compute LogSoftmax  
        std::vector<float> log_softmax_output(num_classes);  
        float max_val = *std::max_element(output_data, output_data + num_classes);  
        float sum_exp = 0.0;  
    
        for (int i = 0; i < num_classes; ++i) {
             sum_exp += std::exp(output_data[i] - max_val);  
        }  
    
        // Compute log of the sum of exponentials  
        float log_sum_exp = std::log(sum_exp);  
    
        // Calculate LogSoftmax for each class  
        for (int i = 0; i < num_classes; ++i) {  
            log_softmax_output[i] = output_data[i] - max_val - log_sum_exp;  
        }  
    
        // Find the predicted class and its confidence  
        int predicted_class = std::distance(log_softmax_output.begin(), std::max_element(log_softmax_output.begin(), log_softmax_output.end()));  
        float confidence = std::exp(log_softmax_output[predicted_class]); // Convert log probability back to probability  
    
        std::cout << "Emotion: " << emotion_array[predicted_class] << " : " << confidence << std::endl;
        delete[] grayscaleData;  
        delete[] resizedData;

        if (predicted_class == 1)
            return 1;
        else
            return 2;

    }
    return 0;
}
