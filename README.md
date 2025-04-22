### Privacy Preserving Computer Vision with Secret Sharing

This repo is designed to recreate my final project for AI for CPS, in Spring 2025.

Due to my use of a library called CrypTen, the code reproducibility will require a small amount of training to recreate the results from my project. 

CrypTen's model saving does not currently work, and due to this I have setup several scripts to train the models automatically, however these are very small models, with just 1-3 layers and are trained on feature extracted data.

The first step will be to ensure that the environment is working correctly. CrypTen is only supported on Mac and Linux, if you are not running either of these operating systems I recommend using a docker image or virtualbox instance for the code. I recommend an ubuntu jammy jellyfish based image, because that is what I have been using. 

If you are running those systems and would like to install CrypTen, and all other dependencies natively, you are only required to run the following command:

pip install -r requirements.txt

This should clone and install the CrypTen library, and all other dependencies, using pip. I *HIGHLY* recommend the use of conda if you go this route. Because of its use of an older version of sklearn, you may have to use the following command:

**export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True**

if installing locally, it is highly important that you set this to false once completed.

##### Creating the feature extracted datasets
Due to space constraints within Github, the data/ directory has not been uploaded to the repository. Luckily, these datasets are very common and available in torchvision, to create the necessary files, simply run 
**python3 create_features.py**
If everything is installed properly it should take only a minute or two and output both cifar10_swin_features.pt and cifar100_swin_features.pt.

##### Running the scripts

Once the environment is working properly, there will be 3 scripts that need to be run. These will take a few minutes to an hour, I highly recommend running them at night or when you have free compute available. The following scripts will each run 10 epochs of training on their respective datasets. The CIFAR-10 and CIFAR-100 scripts will run much faster, given that they are smaller models. The MNIST script will take quite a while. The full training of these models took several hours to converge, but if you compare the training and validation losses/accuracies with the graphs and training_outputs folder, you should see that they line up very well.

python3 MPC_Crypten_cifar10_autograd.py

python3 MPC_Crypten_cifar100_autograd.py

python3 MPC_Crypten_MNIST_autograd.py

If you would like to graph the outputs, there is a supplied graphing.ipynb file. Simply copy and paste the output of each script into a file and change the read path in the second block, then run all parts of the script. 

for information installing and running CrypTen, check its github page, https://github.com/facebookresearch/CrypTen.
