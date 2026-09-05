from deviations import verify as vf

# Trimmed from `cuobjdump -res-usage` on the sm75 image's CUMVS binary: OpenCV's
# cv::cudev kernels at every architecture, the method's own (cv::cuda::gpu) at
# sm_75 only, plus a PTX section that must not count as SASS.
RES_USAGE = """\
Fatbin elf code:
================
arch = sm_50
code version = [1,7]
host = linux
compile_size = 64bit

Resource usage:
 Common:
  GLOBAL:0
 Function _ZN2cv5cudev21grid_transform_detail14transformSimpleIiiNS_3cuda6device10binder2ndINS0_9saturate_castIicEEEEEEvNS0_10PtrStepSzIT_EENS7_IT0_EET1_:
  REG:10 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:360 TEXTURE:0 SURFACE:0 SAMPLER:0

Fatbin ptx code:
================
arch = sm_75
code version = [8,7]
host = linux
compile_size = 64bit
identifier = /sota/cuda-multi-view-stereo/modules/cuda_multi_view_stereo/src/cuda_multi_view_stereo.cu
ptxasOptions =  --generate-line-info

Fatbin elf code:
================
arch = sm_75
code version = [1,7]
host = linux
compile_size = 64bit

Resource usage:
 Common:
  GLOBAL:24 CONSTANT[3]:2704 CONSTANT[4]:8
 Function _ZN2cv4cuda3gpu32initializeDepthsAndNormalsKernelENS0_9PtrStepSzIfEENS2_INS1_4Vec_IfLi3EEEEEff:
  REG:24 STACK:32 SHARED:0 LOCAL:0 CONSTANT[2]:8 CONSTANT[0]:408 TEXTURE:0 SURFACE:0 SAMPLER:0
 Function _ZN2cv5cudev21grid_transform_detail14transformSimpleIiiNS_3cuda6device10binder2ndINS0_9saturate_castIicEEEEEEvNS0_10PtrStepSzIT_EENS7_IT0_EET1_:
  REG:10 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:360 TEXTURE:0 SURFACE:0 SAMPLER:0

Fatbin elf code:
================
arch = sm_120
code version = [1,7]
host = linux
compile_size = 64bit

Resource usage:
 Common:
  GLOBAL:0
 Function _ZN2cv5cudev15integral_detail22horizontal_pass_8u_shflEPKhPjm:
  REG:20 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:376 TEXTURE:0 SURFACE:0 SAMPLER:0
"""


def test_kernels_are_grouped_by_sass_architecture():
    kernels = vf.kernels_by_architecture(RES_USAGE)
    assert set(kernels) == {"sm_50", "sm_75", "sm_120"}
    assert len(kernels["sm_75"]) == 2


def test_method_kernels_are_told_apart_from_opencv():
    own, opencv = vf.split_method_kernels(vf.kernels_by_architecture(RES_USAGE))
    assert own == {"sm_75"}
    assert opencv == {"sm_50", "sm_75", "sm_120"}


def test_plain_c_kernels_count_as_method_kernels():
    text = RES_USAGE.replace("_ZN2cv4cuda3gpu32initializeDepthsAndNormalsKernel", "_Z14RedPixelFilterPK6CameraP6float4Pf")
    own, _ = vf.split_method_kernels(vf.kernels_by_architecture(text))
    assert own == {"sm_75"}
