# GPU-Architecture-and-Computing-labs

<img src="https://media3.giphy.com/media/v1.Y2lkPTc5MGI3NjExd2tkMDhlbHh1NjVveTYxMHllenpob3lzcW1seWFvbDY2ZnBleHR2cyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/xTiJ4cVWew0klLuY96/giphy.gif"/>

    This repository is for my GPU Architecture and Computing Labs.

## <img align= center width=50px height=50px src="https://user-images.githubusercontent.com/71986226/154075883-2a5679d2-b411-448f-b423-9565babf35aa.gif"> Table of Contents
- <a href ="#Overview">Overview</a>
- <a href ="#started"> Get Started</a>
- <a href ="#contributors">Contributors</a>
- <a href ="#license">License</a>

## <img align="center"  height =50px src="https://user-images.githubusercontent.com/71986226/154076110-1233d7a8-92c2-4d79-82c1-30e278aa518a.gif"> Project Overview <a id = "Overview"></a>

## Lab 1

### C Refresher and Environment Setup

- To refresh the C programming language.
- To setup the environment for CUDA programming.
- Lab was to make code in c to accept a 2D array and returns the sum of the numbers formed by concatenating the elements(non negative) of each column.

  ![lab1_0](images/lab1_0.png)
  
  E.g. the given matrix returns 51362
  Explanation: ‘1052 + 20104 + 30206 = 51362’
  Input format: command line arguments as:
  nrows ncols nrows*ncols numbers
  For this matrix: 3 3 10 20 30 5 10 20 2 4 6
  The program should print 51362, nothing more. It must be compiled through nvcc successfully.

## Lab 2

### Matrix addition and matrix-vector multiplication

- Complete the provided matrix addition example, following these cases:

  A.   kernel1: each thread produces one output matrix element
  B.   kernel2: each thread produces one output matrix row
  C.   kernel3: each thread produces one output matrix column
  Analyze the pros and cons of each of the kernels above by using nvprof with large matrix sizes to validate your points. Collect your insights in a PDF report and explain them.

- Implement a matrix–vector multiplication kernel. Use one thread to calculate an output vector element.

Let both programs read testcases from a .txt file and print the output to another. Their pathes are to be provided as command line arguments. Sample test file and invoking command are to be attached to the e-learning page.

## <img  align= center width=50px height=50px src="https://c.tenor.com/HgX89Yku5V4AAAAi/to-the-moon.gif"> Get Started <a id = "started"></a>

To get started with the project, follow these steps:

1. Clone the repository to your local machine.
2. Run the provided code.
3. Customize the code and add any additional features as needed.
4. Enjoy.

<hr style="background-color: #4b4c60"></hr>
<a id ="Contributors"></a>

## <img align="center"  height =60px src="https://user-images.githubusercontent.com/63050133/156777293-72a6e681-2582-4a9d-ad92-09d1181d47c7.gif"> Contributors <a id ="contributors"></a>

<br>
<table >
  <tr>
        <td align="center"><a href="https://github.com/Ahmed-H300"><img src="https://avatars.githubusercontent.com/u/67925988?v=4" width="150px;" alt=""/><br /><sub><b>Ahmed Hany</b></sub></a><br /></td>
  </tr>
  <tr>
        <td align="center"><a href="https://github.com/ahmedmadbouly186"><img src="https://avatars.githubusercontent.com/u/66012617?v=4" width="150px;" alt=""/><br /><sub><b>Ahmed Madbouly</b></sub></a><br /></td>
  </tr>
</table>

<hr style="background-color: #4b4c60"></hr>

## <img  align= center width=50px height=50px src="https://media1.giphy.com/media/ggoKD4cFbqd4nyugH2/giphy.gif?cid=6c09b9527jpi8kfxsj6eswuvb7ay2p0rgv57b7wg0jkihhhv&rid=giphy.gif&ct=s"> License <a id = "license"></a>
This software is licensed under MIT License, See [License](https://github.com/Ahmed-H300/GPU-Architecture-and-Computing-labs/blob/main/LICENSE) for more information ©Ahmed Hany.
