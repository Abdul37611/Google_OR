# Installation Instructions

Supports python 3.7+ on Linux/MacOS systems  
Supports python 3.7-3.9 on Windows - Version limited due to fiona being simpler to install into a conda environment.

For Unix systems

```
pip install -r requirements.txt
python setup.py build_ext --inplace
python setup.py install
```

For Windows systems.
Instructions assume Conda is available.
Creates a new environment called env (feel free to use another name).

```
conda create -n env python=3.8 fiona
conda activate env
pip install -r requirements.txt
python setup.py build_ext --inplace
python setup.py install
```

# Running the web app

Navigate to the `visualization` folder

Run the command:

```
streamlit run app.py
```

A browser should open running the web app.