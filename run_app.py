import subprocess
subprocess.run(["pip", "install", "matplotlib==3.9.2", "pandas==2.2.3", "numpy==1.24.3", "scipy==1.10.0", "streamlit==1.44.1"])
subprocess.run(["streamlit", "run", "EEG_Task_NeuroChallenge.py"])