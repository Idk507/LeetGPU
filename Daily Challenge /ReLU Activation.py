"""
Implement a program that performs the Rectified Linear Unit (ReLU) activation function on a vector of 32-bit floating point numbers. The ReLU function sets all negative values to zero and leaves positive values unchanged:

Implementation Requirements
External libraries are not permitted
The solve function signature must remain unchanged
The final result must be stored in output
Example 1:
Input:  input = [-2.0, -1.0, 0.0, 1.0, 2.0]
Output: output = [0.0, 0.0, 0.0, 1.0, 2.0]
Example 2:
Input:  input = [-3.5, 0.0, 4.2]
Output: output = [0.0, 0.0, 4.2]
Constraints
1 ≤ N ≤ 100,000,000

"""
import torch

# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    output.copy_(torch.clamp(input,min=0))
