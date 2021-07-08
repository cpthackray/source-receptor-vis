# source-receptor-vis

Interactive source-receptor relationship visualization for atmospheric PFAS.

<img src="static/ex_web_pfas.png" alt="web app example" width="500"/>

# Prerequisites
 - Flask
 - WTForms
 - cartopy
 - numpy
 - matplotlib
 - xarray


# Post-installation setup

1. Download data (TBA)
2. In a file called myconfig.py, save the following definitions:
 ```python
 SECRET_KEY = "PUT YOUR SECRET STRING HERE"
 DATAPATH = "PATH/TO/THE/DOWNLOADED/DATA/"
 ```
