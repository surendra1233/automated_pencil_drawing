#!/bin/bash

pip install -r requirements.txt

if [ $? -ne 0 ]; then
	echo "Failed to install requirements, please make sure you are connected to the internet and you have pip."
	exit 1
fi


cd scipy-lic
python setup.py install
 
if [ $? -ne 0 ]; then
	echo ""
	echo "Automated Script Failed. Please procede through manual setup"
	exit 1
fi

cd ..
cd vecplot
python setup.py install

if [ $? -ne 0 ]; then
	echo ""
	echo "Automated Script Failed. Please procede through manual setup"
	exit 1
fi

echo "Setup Success."
echo "To run the algorithm on a image please execute `run.sh [image path]`"