Install the python package by the following steps. (If you use python3, replace the GBDT-PL/python with GBDT-PL/python3).
.. code:: sh
    
    git clone https://github.com/GBDT-PL/GBDT-PL.git
    
    cd GBDT-PL/python
    
    pip install --user .

After your installation, you should successfully run
.. code:: python

    import gbdtpl
If when you are using the package, such error occurs
.. code:: sh

    libiomp5.so: cannot open shared object file: No such file or directory
You need to find out the location of libiomp5.so on your machine and append the directory to LD_LIBRARY_PATH environment variable.
