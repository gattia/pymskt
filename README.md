# pyMSKT (Musculoskeletal Toolkit)

pyMSKT is an open-source library for performing quantitative analyses of the musculoskeletal system. It enables creation of surface meshes of musculoskeletal anatomy and then processes these meshes to get quantitative outcomes and visualizatons, like for cartilage thickness.  

<p align="center">
<img src="./images/whole_knee_1.png" width="300">
</p>

# Installation

1. Clone repository: <br>
    ```bash
    git clone https://github.com/gattia/pymskt.git
    ```
2. Move into repository directory: <br>
    ```bash
    cd pymskt
    ```
3. Install dependencies: <br>
    - Best: <br>
        ```bash
        conda env create -f environment.yml
        conda activate mskt
        ```
    - Second - ideally create a virtual environment first & then: <br>
        ```bash
        pip install -r requirements.txt
        ```
3. Install pacakge<br>
    ```bash
    python setup.py install
    ```

### To install itkwidgets (for visualization): 
https://pypi.org/project/itkwidgets/

```bash
conda install -c conda-forge itkwidgets
```

### OR

```bash
pip install itkwidgets
```

If you are using jupyterlab instead of jupyter notebook, you also need to install an extension: 

```
jupyter labextension install @jupyter-widgets/jupyterlab-manager jupyter-matplotlib jupyterlab-datawidgets itkwidgets
```

# Examples
There are jupyter notebook examples in the directory `/examples`

pyMSKT allows you to easily create bone meshes and attribute cartilage to the bone for calculating quantitative outcomes. 

```python
femur = BoneMesh(path_seg_image=location_seg,  # path to the segmentation image being used.
                 label_idx=5,                  # what is the label of this bone.
                 list_cartilage_labels=[1]) # labels for cartilage associted with bone.   
# Create the bone mesh
femur.create_mesh()
# Calcualte cartialge thickness for the cartialge meshes associated with the bone
femur.calc_cartilage_thickness()
femur.save_mesh(os.path.expanduser'~/Downloads/femur.vtk')
```
The saved file can be viewed in many mesh viewers such as [3D Slicer](https://www.slicer.org/) or [Paraview](https://www.paraview.org/). Or, better yet they can be viewed in your jupyter notebook using [itkwidgets](https://pypi.org/project/itkwidgets/): 
```python
from itkwidgets import view

view(geometries=[femur.mesh])
```

![](/images/femur_itkwidgets.png)



# Tests
- Run tests using `pytest` in the home directory or `make test`
- Can get coverage statistics by running: 
    - `coverage run -m pytest`
    or if using make: 
    - `make coverage`

- When updating the cython code, it is not re-built when we re-install. Therefore we force it to do this: 
    - `python setup.py build_ext -i --force`          

# Contributing

### Tests
If you add a new function, or functionality to `pymskt` please add appropriate tests as well. 
The tests are located in `/testing` and are organized as: 
`/testing/[pymskt_submodule]/[python_filename_being_tested]/[name_of_function_being_tested]_test.py`

The tests use `pytest`. If you are not familiar with `pytest` a brief example is provided [here](https://docs.pytest.org/en/6.2.x/getting-started.html). 

Currently, multiple tests are not passing, this is because dummy tests have been created where a test should go. If you want to help but dont know how or where to start, filling in these tests would be a great place to start. 