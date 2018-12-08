cd ..
rm -r build 
rm -r lib
cmake -H. -Bbuild
cmake --build build -- -j VERBOSE=1
cd python
pip uninstall gbdtpl
pip install --user .
python test.py
