

## 1. Install Packages

```bash
pip install -r requirements.txt
```

## 2. Start Demo:

Starting all Eagle2-1B/2B/9B/34B demos on 8xA100.

```bash
bash start_demo.sh
```

otherwise you can choose to start one model with:

```bash
python controller.py &
python model_worker.py --model-path nvidia/Eagle2-1B \
--model-name Eagle-1B  --port 6213 --worker-address http://127.0.0.1:6213 &
streamlit run app.py 
```

## Troubleshooting

- issue:
```bash
File "Eagle2/streamlit_demo/app.py", line 23, in <module>
    import cv2
File "/opt/conda/lib/python3.11/site-packages/cv2/__init__.py", line 181, in <module>
    bootstrap()
File "/opt/conda/lib/python3.11/site-packages/cv2/__init__.py", line 153, in bootstrap
    native_module = importlib.import_module("cv2")
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/opt/conda/lib/python3.11/importlib/__init__.py", line 126, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
```
- solution:
```bash
sudo apt-get update
sudo apt-get install libgl1-mesa-glx
sudo apt-get install libgl1-mesa-dev
```