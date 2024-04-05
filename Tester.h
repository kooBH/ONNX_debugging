#pragma once
#include <onnxruntime_cxx_api.h>
#include <algorithm>
#include <vector>
#include <stdlib.h>
#include <stdio.h>

class Tester{
private:
	std::vector<std::vector<int64_t>> input_node_dims;
	std::vector<std::vector<int64_t>> output_node_dims;
	std::vector<std::vector<float>> input_node_vectors;
	std::vector<std::vector<float>> output_node_vectors;
	std::vector<size_t> input_sizes;
	std::vector<size_t> output_sizes;
	std::vector<Ort::Value> inputTensors;
	std::vector<Ort::Value> outputTensors;
	size_t num_input_nodes;
	size_t num_output_nodes;

	std::vector<const char *> input_node_names = {"input"};
	std::vector<const char *> output_node_names = {"output"};
	Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "Tester");
	Ort::SessionOptions session_options;
	Ort::Session *session;


public:
	Tester();
	void Process(std::vector<float> data_in,const char* path_this);
	void Validate(const char* path_this, const char* path_that);
};

