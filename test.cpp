#include "Tester.h"


int main(){
    printf("Init model\n");
    Tester t;

    // Read data and run

    // Read file "input.bin" into vector of float data
    printf("Read input\n");
    std::vector<float> data(10*5*5);
    FILE* f = fopen("../input.bin","rb");
    fread(data.data(),sizeof(float),data.size(),f);
    fclose(f);

    // void Process(float* data,const char* path_this);
    printf("Run ONNX\n");
    t.Process(data, "../output_cpp.bin");
    printf("Validate\n");
    t.Validate("../output_cpp.bin","../output_python.bin");


    return 0;
}
