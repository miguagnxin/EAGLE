python controller.py &
CUDA_VISIBLE_DEVICES=1 python model_worker.py --model-path nvidia/Eagle2-9B \
--model-name Eagle-9B  --port 6212 --worker-address http://127.0.0.1:6212 &
CUDA_VISIBLE_DEVICES=2 python model_worker.py --model-path nvidia/Eagle2-1B \
--model-name Eagle-1B  --port 6213 --worker-address http://127.0.0.1:6213 &
CUDA_VISIBLE_DEVICES=3 python model_worker.py --model-path nvidia/Eagle2-2B \
--model-name Eagle-2B  --port 6214 --worker-address http://127.0.0.1:6214 &
streamlit run app.py 

