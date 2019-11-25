## automated_pencil_art

A python implemenation of the research paper [[1](#paper)]

### Automated Setup

If you have bash, then you can just run the script `install.sh` to setup the environment and then run `run.sh` for running the algorithm. If automated run is failing for you, then go with manual setup
```shell
# giving permissions for execution
chmod +x install.sh
chmod +x run.sh
```
```shell
./install.sh             # setups the environment
./run.sh [image_path]    # runs the algorithm on given image
```

### Manual Setup

**To Run:**

Install required packages

```shell
pip install -r requirements.txt
```

Also go to `scipy-lic`

```shell
cd scipy-lic/
python setup.py install
# and
cd vecplot/
python setup.py install
```

#### To run an example

```shell
cd src/
python final.py <IMAGE_PATH>
```

Open `jupyter notebook` only from `src` if needs to view a notebook.

**Slightly detailed example in** `src/main.ipynb`

### Sample Outputs 


##### paper

[1] Automatic generation of accentuated pencil drawing with saliency
map and LIC. [url](https://www.researchgate.net/publication/235197579_Automatic_generation_of_accentuated_Pencil_Drawing_with_Saliency_Map_and_LIC)
