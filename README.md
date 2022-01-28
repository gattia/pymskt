# Installation

1. Clone repository: 
    `git clone https://github.com/gattia/pymskt.git`
2. Move into repository directory: 
    `cd pymskt`
3. Install dependencies: 
    Best: 
        `conda env create -f environment.yml`
    Second - ideally create a virtual environment first & then: 
        `pip install -r requirements.txt`
3. Install pacakge
    `python setup.py install`



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