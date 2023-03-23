#include <C:/Users/5D623CA2626A/Downloads/libtorch/include/torch/script.h>
#include <C:/Users/5D623CA2626A/Downloads/libtorch/include/torch/torch.h>
#include <iostream>

int main() {
  // Load the TorchScript model
  torch::jit::script::Module module;
  try {
    module = torch::jit::load("CNNModel.ptc");
  }
  catch (const c10::Error& e) {
    std::cerr << "Error loading the model\n";
    return -1;
  }

  // Set the module to evaluation mode
  module.eval();

  // Load the input image
  std::string image_path = "Airplane_Img.jpg";
  auto input_image = torch::data::datasets::ImageFolder::read_image(image_path);

  // Resize and normalize the input image
  input_image = torch::data::transforms::Resize({32, 32})(input_image);
  input_image = torch::data::transforms::Normalize({0.5, 0.5, 0.5}, {0.5, 0.5, 0.5})(input_image);

  // Convert the input image to a tensor
  auto input_tensor = torch::unsqueeze(input_image, 0);
  input_tensor = input_tensor.permute({0, 3, 1, 2});
  input_tensor = input_tensor.to(torch::kFloat);

  // Run inference on the input tensor
  auto output_tensor = module.forward({input_tensor}).toTensor();

  // Get the predicted class label
  auto output = output_tensor.squeeze();
  auto max_result = output.max(0);
  auto predicted_label = std::get<1>(max_result).item<int>();

  // Print the predicted label
  std::cout << "Predicted label: " << predicted_label << std::endl;

  return 0;
}