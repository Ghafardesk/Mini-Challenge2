## AMD OCR Software
hello everyone, im building an OCR software for AMD AI Academy challenge.

## Tutorial
   follow this steps for installing depoendencies, models and running some test

   1. install dependencies
     <script>python -m pip install --upgrade pip</script>
     <script>python -m install requirements.txt</script>

   2. set model cache to /models directory

    <script>mkdir -p /models</script>
    <script>export HF_HOME=/models</script>
    <script>export TRANSFORMERS_CACHE=/models</script>
    <script>export TORCH_HOME=/models</script>

    3. download models

     depende on device you used, this model can installed locally if you have monstrous GPU,  if dont, just open amd notebook and clone this repo then install and set required dependency above
    <script>python download_model.py</script>