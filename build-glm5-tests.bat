@echo off
rem Build the glm5 GPU kernel tests with nvcc (sm_120 = RTX 5090) into build-glm-gpu\  (2026-10-08)
rem Each test links the Strata kernel files it checks.
setlocal
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set NVCC="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\nvcc.exe"
set R=%~dp0
set K=%R%src\kernels\cuda
if not exist "%R%build-glm-gpu" mkdir "%R%build-glm-gpu"
call :one glm5_mhc_test "%K%\fused_mhc.cu" || exit /b 1
call :one glm5_router_test "%K%\router_top10.cu" || exit /b 1
call :one glm5_ffn_test "%K%\verify_kernels.cu" "%K%\shared_expert.cu" "%K%\native_mmvq.cu" "%K%\iq_kernels.cu" "%K%\native_bf16.cu" "%K%\quantize_act.cu" "%K%\s_gemv.cu" "%K%\s2_gemv_q8.cu" || exit /b 1
call :one glm5_kda_test "%K%\verify_kernels.cu" || exit /b 1
call :one glm5_mla_test "%K%\qsa_decode_attn.cu" || exit /b 1
call :one glm5_kpool_test "%K%\native_qsa_indexer.cu" "%K%\native_rope.cu" || exit /b 1
call :one glm5_layer_test "%K%\fused_mhc.cu" "%K%\verify_kernels.cu" "%K%\shared_expert.cu" "%K%\native_mmvq.cu" "%K%\iq_kernels.cu" "%K%\native_bf16.cu" "%K%\quantize_act.cu" "%K%\s_gemv.cu" "%K%\s2_gemv_q8.cu" || exit /b 1
call :one glm5_batch_test "%K%\glm5_misc.cu" "%K%\fused_mhc.cu" "%K%\verify_kernels.cu" "%K%\shared_expert.cu" "%K%\native_mmvq.cu" "%K%\iq_kernels.cu" "%K%\native_bf16.cu" "%K%\quantize_act.cu" "%K%\s_gemv.cu" "%K%\s2_gemv_q8.cu" || exit /b 1
exit /b 0

:one
set NAME=%~1
set FILES=
:more
shift
if "%~1"=="" goto go
set FILES=%FILES% "%~1"
goto more
:go
echo [build] %NAME%
%NVCC% -O3 -std=c++20 -arch=sm_120 -I"%R%include" -I"%R%third_party\ggml" "%R%tests\cuda\%NAME%.cu" %FILES% -o "%R%build-glm-gpu\%NAME%.exe" || exit /b 1
exit /b 0
