"""
Write a program that transposes a matrix of 32-bit floating point numbers on a GPU. The transpose of a matrix switches its rows and columns. Given a matrix 
 of dimensions 
, the transpose 
 will have dimensions 
. All matrices are stored in row-major format.

Implementation Requirements
Use only native features (external libraries are not permitted)
The solve function signature must remain unchanged
The final result must be stored in the matrix output
Example 1:
Input: 2×3 matrix

 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
Output: 3×2 matrix

 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
Example 2:
Input: 3×1 matrix

 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
Output: 1×3 matrix

 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
Constraints
1 ≤ rows, cols ≤ 8192
Input matrix dimensions: rows × cols
Output matrix dimensions: cols × rows

"""
import torch

# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, rows: int, cols: int):
    output.copy_(input.t())
