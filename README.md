## Implementation of state-of-the-art Deep Reinforcement Learning Algorithms for Continuous Action Spaces

```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
export PYTHONPATH=$(pwd)
```

#### Installing PROJ version 8 (requirement for movingpandas)
```bash
 mkdir ~/GPS
 cd ~/GPS
 wget https://download.osgeo.org/proj/proj-8.0.0.tar.gz
 tar xzvf proj-8.0.0.tar.gz

 cd proj-8.0.0

 mkdir build
 cd build

 ccmake .. -DCMAKE_INSTALL_PREFIX=/usr
 cmake --build . -j2
 sudo cmake --build . --target install
 ```

 ### Install GEOS (needs ubuntu 20.04+)
 ```bash
 sudo apt update
 sudo apt install libgeos++-dev libgeos-3.8.0 libgeos-c1v5 libgeos-dev libgeos-doc
 ```