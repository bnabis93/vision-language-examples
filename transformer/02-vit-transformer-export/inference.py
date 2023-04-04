import time

import numpy as np
import onnxruntime as rt

# Get the available providers
options = rt.SessionOptions()
print(rt.get_available_providers())

# Load the ONNX model
sess = rt.InferenceSession("onnx/model.onnx", providers=rt.get_available_providers())

# Get the input and output names
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name
print(f"Input name: {input_name}, output name: {output_name}")

# Create a random input tensor
input_tensor = np.random.randn(1, 3, 224, 224).astype(np.float32)

# Warm-up
output = sess.run([output_name], {input_name: input_tensor})

# Run the model and measure the inference time
times = []
for _ in range(100):
    start_time = time.time()
    output = sess.run([output_name], {input_name: input_tensor})
    end_time = time.time()
    times.append(end_time - start_time)

avg_inference_time = np.mean(times)
print(f"Average Inference time: {avg_inference_time:.6f} seconds")
