cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(survos_cuda_project LANGUAGES CXX CUDA)
 
add_library(survos_cuda SHARED
						cuda.cu 
						tv.cu
						diffusion.cu
						chambolle2005.cu
						chambolle2011.cu
						bregman.cu
						slic.cu
						convolutions_raw.cu
						convolutions_separable.cu
						convolutions_separable_shared.cu
						symmetric_eigvals3S.cu
  )
include_directories(${PROJECT_SOURCE_DIR})
target_compile_features(survos_cuda PUBLIC cxx_std_11)
 
set_target_properties( survos_cuda
					   PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
install(TARGETS survos_cuda 
		LIBRARY DESTINATION "${INSTALL_LIB_DIR}" COMPONENT lib
		RUNTIME  DESTINATION "${INSTALL_BIN_DIR}" COMPONENT bin
		ARCHIVE  DESTINATION "${INSTALL_LIB_DIR}" COMPONENT lib)