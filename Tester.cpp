#include "Tester.h"

Tester::Tester(){
	session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
	session = new Ort::Session(env,"../export.onnx", session_options);

	num_input_nodes = session->GetInputCount();
	num_output_nodes = session->GetOutputCount();

	int idx = 0;
	printf("init input nodes : %d\n",num_input_nodes);	
	for (int i = 0; i < num_input_nodes; i++) {
		Ort::TypeInfo inputTypeInfo = session->GetInputTypeInfo(i);
		auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
		std::vector<int64_t> inputDims = inputTensorInfo.GetShape();
		input_node_dims.push_back(inputDims);
		
		//calculate input_tensor size
		size_t input_tensor_size = 1;
		idx = 0;
		for (auto dim : inputDims) {
			printf("dim[%d] : %d \n",idx++, dim);
			input_tensor_size *= dim;
		}
		printf("input tensor size[%d] : %d \n",i,input_tensor_size);
		//allocate empty vector
		std::vector<float> input_tensor_values(input_tensor_size, 0);
		input_node_vectors.push_back(input_tensor_values);
		input_sizes.push_back(input_tensor_size);
		inputTensors.push_back(Ort::Value::CreateTensor<float>(
			memoryInfo, input_node_vectors[i].data(), input_sizes[i], input_node_dims[i].data(),
			input_node_dims[i].size()));
	}
	printf("init output nodes : %d\n",num_output_nodes);	
	// initialize output tensors
	for (int i = 0; i < num_output_nodes; i++) {
		Ort::TypeInfo outputTypeInfo = session->GetOutputTypeInfo(i);
		auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
		std::vector<int64_t> outputDims = outputTensorInfo.GetShape();

		output_node_dims.push_back(outputDims);

		//calculate input_tensor size
		idx = 0;
		size_t output_tensor_size = 1;
		for (auto dim : outputDims) {
			output_tensor_size *= dim;
			printf("dim[%d] : %d \n",i, dim);
		}
		printf("output tensor size[%d] : %d \n",i,output_tensor_size);
		//allocate empty vector
		std::vector<float> output_tensor_values(output_tensor_size, 0);
		output_node_vectors.push_back(output_tensor_values);
		output_sizes.push_back(output_tensor_size);
		outputTensors.push_back(Ort::Value::CreateTensor<float>(
			memoryInfo, output_node_vectors[i].data(), output_sizes[i],
			output_node_dims[i].data(), output_node_dims[i].size()));
	}
}

void Tester::Process(std::vector<float> data_in,const char* path_this) {

	std::vector<float>vect_out(10);

	input_node_vectors[0] = data_in;
	Ort::RunOptions run_options;
	session->Run(Ort::RunOptions{ nullptr }, input_node_names.data(), inputTensors.data(), inputTensors.size(), output_node_names.data(), outputTensors.data(), outputTensors.size());
	vect_out = output_node_vectors[0];

	// save vect out
	FILE *f = fopen(path_this, "wb");
	fwrite(vect_out.data(), sizeof(float), vect_out.size(), f);
	fclose(f);
}

void Tester::Validate(const char* path_this, const char* path_that){
	FILE *f_this = fopen(path_this, "rb");
	FILE *f_that = fopen(path_that, "rb");

	// compare both files
	float this_val, that_val;
	float error = 0;
	while (fread(&this_val, sizeof(float), 1, f_this) == 1 && fread(&that_val, sizeof(float), 1, f_that) == 1) {
		printf("%f %f\n",this_val, that_val);
		error += abs(this_val - that_val);
	}
	fclose(f_this);
	fclose(f_that);

	printf("ERROR of %s and %s : %f\n",path_this, path_that, error);
}

