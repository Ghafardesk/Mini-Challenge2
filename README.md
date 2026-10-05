## AMD OCR Software
hello everyone, im building an OCR software for AMD AI Academy challenge.

## Tutorial
   follow this steps for installing depoendencies, models and running some test

   1. install dependencies
     
    python -m pip install --upgrade pip
    python -m install requirements.txt

   2. set model cache to /models directory

    mkdir -p /models
    export HF_HOME=/models
    export TRANSFORMERS_CACHE=/models
    export TORCH_HOME=/models

    3. download models

    depend on device you used, this model can installed locally if you have 
    monstrous GPU,  if dont, just open amd notebook and clone this repo then
    install and set required dependency above
    
    python download_model.py

    if fail and huggingface need login, run this and input your hf token
    ensure you are approved for using this model
    
    huggingface-cli login