# Introduction to CUDA

## Notes

- ***corssplatform compilers***: compile if you don't have NVIDIA card on your system

- the green in picture-1 (notebook of lab 2) repersent cores

- you should focus on parallism to get benifit of the GPU

- SM --> streaming microprocessor

- kol mgmo3a mn el cores mgmo3a fe SMs

- Warp --> 32 core combined

- SM --> has group of warps

- grid --> SM --> warps     --> cores(HW)
                            --> threads(SW)

- grid hwa lyaer in GPU

- block can be 1D, 2D or 3D

- threads in the block can be syncronized

- dont assume order of exc in blocks

- but there is a way to synchronzie by making clusters

- nvprof --> profiling tool

- profiling tool --> explain the time taken by the GPU any activity related to GPU

- NOTE: nv estands for NVIDIA

- ! -->- execute Linux command

- nvprof takes object file --> !NVCC p.cu -o out

- out is the object file

- why DTH is bigger

  - bec memcpy is syn
  - pci ely byn2l
  - data not in the same place

- the kernel is blocking or not?

  - async

- cuda memcopy is sync so it will stop till kernel is finished

- there is a way to make kernel syn or make memcpy aSync

- Dim of Block best practice to be the Dim of the structure of the data to help the striding to be simpler

- max is 1024 thread for the single block and limit of blocks is related to specs of GPU

- best practice 16 * 16 block = 256 threads
