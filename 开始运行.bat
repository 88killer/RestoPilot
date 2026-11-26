@echo off
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
set TORCH_HOME=%CD%\AgenticIR\work\torch_cache
set OMP_NUM_THREADS=4
set MKL_NUM_THREADS=4
if not exist ".venv\setup.complete" (
  echo Creating Python environment. Python 3.11 and internet are required.
  py -3.11 -m venv .venv || goto :error
  .venv\Scripts\python.exe -m pip install --upgrade pip || goto :error
  .venv\Scripts\python.exe -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128 || goto :error
  .venv\Scripts\python.exe -m pip install -r requirements.txt || goto :error
  echo complete> .venv\setup.complete
)
.venv\Scripts\python.exe -c "import torch; assert torch.cuda.is_available(), 'NVIDIA CUDA GPU is required'; print('GPU:',torch.cuda.get_device_name(0))" || goto :error
if not exist "data.ready" (
  .venv\Scripts\python.exe generate_data.py || goto :error
  .venv\Scripts\python.exe prepare_manifest.py || goto :error
  echo complete> data.ready
)
.venv\Scripts\python.exe run_best.py . || goto :error
echo Done. Results are in the output folder.
pause
exit /b 0
:error
echo Failed. Please copy the error message and send it to the project owner.
pause
exit /b 1
