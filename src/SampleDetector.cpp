#include <sys/stat.h>
#include <fstream>
#include <glog/logging.h>

#include "SampleDetector.h"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "ji_utils.h"
#include "./logging.h"
#define INPUT_NAME "images"
#define OUTPUT_NAME "output"
using namespace nvinfer1;


static bool ifFileExists(const char *FileName)
{
    struct stat my_stat;
    return (stat(FileName, &my_stat) == 0);
}

SampleDetector::SampleDetector()
{
    
}

void SampleDetector::loadOnnx(const std::string strModelName)
{
    Logger gLogger;
    //根据tensorrt pipeline 构建网络
    IBuilder* builder = createInferBuilder(gLogger);
    builder->setMaxBatchSize(1);
    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);  
    INetworkDefinition* network = builder->createNetworkV2(explicitBatch);
    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, gLogger);
    parser->parseFromFile(strModelName.c_str(), static_cast<int>(ILogger::Severity::kWARNING));
    IBuilderConfig* config = builder->createBuilderConfig();
    config->setMaxWorkspaceSize(1ULL << 30);    
    m_CudaEngine = builder->buildEngineWithConfig(*network, *config);    

    std::string strTrtName = strModelName;
    size_t sep_pos = strTrtName.find_last_of(".");
    strTrtName = strTrtName.substr(0, sep_pos) + ".trt";
    IHostMemory *gieModelStream = m_CudaEngine->serialize();
    std::string serialize_str;
    std::ofstream serialize_output_stream;
    serialize_str.resize(gieModelStream->size());   
    memcpy((void*)serialize_str.data(),gieModelStream->data(),gieModelStream->size());
    serialize_output_stream.open(strTrtName.c_str());
    serialize_output_stream<<serialize_str;
    serialize_output_stream.close();
    m_CudaContext = m_CudaEngine->createExecutionContext();
    parser->destroy();
    network->destroy();
    config->destroy();
    builder->destroy();
}

void SampleDetector::loadTrt(const std::string strName)
{
    Logger gLogger;
    IRuntime* runtime = createInferRuntime(gLogger);    
    std::ifstream fin(strName);
    std::string cached_engine = "";
    while (fin.peek() != EOF)
    { 
        std::stringstream buffer;
        buffer << fin.rdbuf();
        cached_engine.append(buffer.str());
    }
    fin.close();
    m_CudaEngine = runtime->deserializeCudaEngine(cached_engine.data(), cached_engine.size(), nullptr);
    m_CudaContext = m_CudaEngine->createExecutionContext();
    runtime->destroy();
}

bool SampleDetector::Init(const std::string& strModelName, float thresh)
{
    mThresh = thresh;
    std::string strTrtName = strModelName;
    size_t sep_pos = strTrtName.find_last_of(".");
    strTrtName = strTrtName.substr(0, sep_pos) + ".trt";
    if(ifFileExists(strTrtName.c_str()))
    {        
        loadTrt(strTrtName);
    }
    else
    {
        loadOnnx(strModelName);
    }    
    // 分配输入输出的空间,DEVICE侧和HOST侧
    m_iInputIndex = m_CudaEngine->getBindingIndex(INPUT_NAME);
    m_iOutputIndex = m_CudaEngine->getBindingIndex(OUTPUT_NAME);     

    Dims dims_i = m_CudaEngine->getBindingDimensions(m_iInputIndex);
    SDKLOG(INFO) << "input dims " << dims_i.d[0] << " " << dims_i.d[1] << " " << dims_i.d[2] << " " << dims_i.d[3];
    int size = dims_i.d[0] * dims_i.d[1] * dims_i.d[2] * dims_i.d[3];
    
    m_InputSize = cv::Size(dims_i.d[3], dims_i.d[2]);

    cudaMalloc(&m_ArrayDevMemory[m_iInputIndex], size * sizeof(float));
    m_ArrayHostMemory[m_iInputIndex] = malloc(size * sizeof(float));
    //方便NHWC到NCHW的预处理
    m_InputWrappers.emplace_back(dims_i.d[2], dims_i.d[3], CV_32FC1, m_ArrayHostMemory[m_iInputIndex]);
    m_InputWrappers.emplace_back(dims_i.d[2], dims_i.d[3], CV_32FC1, m_ArrayHostMemory[m_iInputIndex] + sizeof(float) * dims_i.d[2] * dims_i.d[3] );
    m_InputWrappers.emplace_back(dims_i.d[2], dims_i.d[3], CV_32FC1, m_ArrayHostMemory[m_iInputIndex] + 2 * sizeof(float) * dims_i.d[2] * dims_i.d[3]);
    m_ArraySize[m_iInputIndex] = size *sizeof(float);
    dims_i = m_CudaEngine->getBindingDimensions(m_iOutputIndex);
    SDKLOG(INFO) << "output dims "<< dims_i.nbDims << " " << dims_i.d[0] << " " << dims_i.d[1] << " " << dims_i.d[2];    
    size = dims_i.d[0] * dims_i.d[1] * dims_i.d[2];
    m_iClassNums = dims_i.d[2] - 5;
    cudaMalloc(&m_ArrayDevMemory[m_iOutputIndex], size * sizeof(float));
    m_ArrayHostMemory[m_iOutputIndex] = malloc( size * sizeof(float));
    m_ArraySize[m_iOutputIndex] = size *sizeof(float);
    cudaStreamCreate(&m_CudaStream);    
    m_bUninit = false;
}

bool SampleDetector::UnInit()
{
    if(m_bUninit == true)
    {
        return false;
    }
    for(auto &p: m_ArrayDevMemory)
    {      
        cudaFree(p);
        p = nullptr;            
    }        
    for(auto &p: m_ArrayHostMemory)
    {        
        free(p);
        p = nullptr;        
    }        
    cudaStreamDestroy(m_CudaStream);
    m_CudaContext->destroy();
    m_CudaEngine->destroy();
    m_bUninit = true;
}

SampleDetector::~SampleDetector()
{
    UnInit();   
}

bool SampleDetector::ProcessImage(const cv::Mat& img, std::vector<BoxInfo>& DetObjs, float thresh)
{
    mThresh = thresh;
    DetObjs.clear();  
    float r = std::min(m_InputSize.width / static_cast<float>(img.rows), m_InputSize.width / static_cast<float>(img.cols));
    cv::Size new_size = cv::Size{img.cols * r, img.rows * r};    
    cv::Mat tmp_resized;    
    
    cv::resize(img, tmp_resized, new_size);
    m_Resized = cv::Mat( cv::Size(m_InputSize.width, m_InputSize.height), CV_8UC3, cv::Scalar(114, 114, 114));    
    tmp_resized.copyTo(m_Resized(cv::Rect{0, 0, tmp_resized.cols, tmp_resized.rows}));

    
    m_Resized.convertTo(m_Normalized, CV_32FC3);
    cv::split(m_Normalized, m_InputWrappers); 

    auto ret = cudaMemcpyAsync(m_ArrayDevMemory[m_iInputIndex], m_ArrayHostMemory[m_iInputIndex], m_ArraySize[m_iInputIndex], cudaMemcpyHostToDevice, m_CudaStream);
    auto ret1 = m_CudaContext->enqueueV2(m_ArrayDevMemory, m_CudaStream, nullptr);    
    ret = cudaMemcpyAsync(m_ArrayHostMemory[m_iOutputIndex], m_ArrayDevMemory[m_iOutputIndex], m_ArraySize[m_iOutputIndex], cudaMemcpyDeviceToHost, m_CudaStream);
    ret = cudaStreamSynchronize(m_CudaStream);    
    float scale = std::min(m_InputSize.width / (img.cols * 1.0), m_InputSize.height / (img.rows * 1.0));
    decode_outputs((float*)m_ArrayHostMemory[m_iOutputIndex], mThresh, DetObjs, scale, img.cols, img.rows);
}

void SampleDetector::generate_grids_and_stride(std::vector<int>& strides, std::vector<GridAndStride>& grid_strides)
{
    for (auto stride : strides)
    {
        int num_grid_y = m_InputSize.height / stride;
        int num_grid_x = m_InputSize.width / stride;
        for (int g1 = 0; g1 < num_grid_y; g1++)
        {
            for (int g0 = 0; g0 < num_grid_x; g0++)
            {
                grid_strides.push_back((GridAndStride){g0, g1, stride});
            }
        }
    }
}

void SampleDetector::generate_yolox_proposals(std::vector<GridAndStride> grid_strides, float* feat_blob, float prob_threshold, std::vector<BoxInfo>& objects)
{
    const int num_anchors = grid_strides.size();    
    for (int anchor_idx = 0; anchor_idx < num_anchors; anchor_idx++)
    {
        const int grid0 = grid_strides[anchor_idx].grid0;
        const int grid1 = grid_strides[anchor_idx].grid1;
        const int stride = grid_strides[anchor_idx].stride;
        const int basic_pos = anchor_idx * (m_iClassNums + 5);
        float x_center = (feat_blob[basic_pos+0] + grid0) * stride;
        float y_center = (feat_blob[basic_pos+1] + grid1) * stride;
        float w = exp(feat_blob[basic_pos+2]) * stride;
        float h = exp(feat_blob[basic_pos+3]) * stride;
        float x0 = x_center - w * 0.5f;
        float y0 = y_center - h * 0.5f;

        float box_objectness = feat_blob[basic_pos+4];
        for (int class_idx = 0; class_idx < m_iClassNums; class_idx++)
        {
            float box_cls_score = feat_blob[basic_pos + 5 + class_idx];
            float box_prob = box_objectness * box_cls_score;
            if (box_prob > prob_threshold)
            {
                BoxInfo obj;
                obj.x1 = x0;
                obj.y1 = y0;
                obj.x2 = x0 + w;
                obj.y2 = y0 + h;
                obj.label = class_idx;
                obj.score = box_prob;
                objects.push_back(obj);
            }
        }
    }         
    runNms(objects, 0.45);
}

void SampleDetector::runNms(std::vector<BoxInfo>& objects, float thresh) 
{
    auto cmp_lammda = [](const BoxInfo& b1, const BoxInfo& b2){return b1.score < b2.score;};
    std::sort(objects.begin(), objects.end(), cmp_lammda);
    for(int i = 0; i < objects.size(); ++i)
    {
        if( objects[i].score < 0.1 )
        {
            continue;
        }
        for(int j = i + 1; j < objects.size(); ++j)
        {
            cv::Rect rect1 = cv::Rect{objects[i].x1, objects[i].y1, objects[i].x2 - objects[i].x1, objects[i].y2 - objects[i].y1};
            cv::Rect rect2 = cv::Rect{objects[j].x1, objects[j].y1, objects[j].x2 - objects[i].x1, objects[j].y2 - objects[j].y1};
            if(IOU(rect1, rect2) > thresh)   
            {
                objects[i].score = 0.f;
            }
        }
    }
    auto iter = objects.begin();
    while( iter != objects.end() )
    {
        if(iter->score < 0.1)
        {
            iter = objects.erase(iter);
        }
        else
        {
            ++iter;
        }
    }
}

void SampleDetector::decode_outputs(float* prob, float thresh, std::vector<BoxInfo>& objects, float scale, const int img_w, const int img_h) 
{
    std::vector<BoxInfo> proposals;
    std::vector<int> strides = {8, 16, 32};
    std::vector<GridAndStride> grid_strides;
    generate_grids_and_stride(strides, grid_strides);

    generate_yolox_proposals(grid_strides, prob,  thresh, proposals);    

    objects = proposals;    
    for (int i = 0; i < proposals.size(); i++)
    {
        objects[i] = proposals[i];
        float x0 = (objects[i].x1) / scale;
        float y0 = (objects[i].y1) / scale;
        float x1 = (objects[i].x2) / scale;
        float y1 = (objects[i].y2) / scale;
        x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);
        objects[i].x1 = x0;
        objects[i].y1 = y0;
        objects[i].x2 = x1;
        objects[i].y2 = y1;
    }
}