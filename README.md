# ONNX_debugging

1. Python ONNX  
```python torch_model.py```
=> ```ipnut.bin```, ```export.onnx```, ```output_python.bin```  
  
2. C++ ONNX    
```
cd build
cmake ..
make 
./test
```  
=> ```output_cpp.bin``` & Compare with ```output_python.bin```.  