"""
Implement a program that copies an 
 matrix of 32-bit floating point numbers from input array 
 to output array 
 on the GPU. The program should perform a direct element-wise copy so that 
 for all valid indices.

Implementation Requirements
External libraries are not permitted
The solve function signature must remain unchanged
The final result must be stored in matrix B
Example 1:
Input:  A = [[1.0, 2.0],
             [3.0, 4.0]]
Output: B = [[1.0, 2.0],
             [3.0, 4.0]]
Example 2:
Input:  A = [[5.5, 6.6, 7.7],
             [8.8, 9.9, 10.1],
             [11.2, 12.3, 13.4]]
Output: B = [[5.5, 6.6, 7.7],
             [8.8, 9.9, 10.1],
             [11.2, 12.3, 13.4]]
Constraints
1 ≤ N ≤ 4096
All elements are 32-bit floating point numbers

"""
import torch

# A, B are tensors on the GPU
def solve(A: torch.Tensor, B: torch.Tensor, N: int):
    B.copy_(A)
