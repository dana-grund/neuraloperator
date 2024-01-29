====================================
Neural Operator: ensemble extensions
====================================

This branch extends the Fourier Neural Operator to ensemble generation. 

Available Examples
------------------
[describe the shear layer test case]

Available Ensemble Neural Operators
-----------------------------------
[description]

Project setup on the euler cluster at ETH
-----------------------------------------

At installation, create a virtual environment (venv) for Python:

.. code::
  python -m venv --system-site-packages venv-NO-ens

In every new terminal, load the environment modules on euler 
and activate the python environment:

.. code::
  module load gcc/8.2.0 python/3.10.4
  source venv-NO-ens/bin/activate

Install ``neuraloperator``, check out this branch, and its dependencies within the 
venv in developer mode:

.. code::
  cd your_home_dir
  mkdir git
  cd git/
  git clone https://github.com/dana-grund/neuraloperator
  git checkout NO-ensembles
  cd neuraloperator
  pip install -e .
  pip install -r requirements.txt

Test the installation with the quickstart and tests from ``README.rst``.


Working on the euler cluster at ETH
-----------------------------------------

Rule number one: The ETH euler wiki knows everything 
you need, mainly here: https://scicomp.ethz.ch/wiki/Using_the_batch_system

The easiest access to Euler without further setup needed 
is via the JupyterHub, https://scicomp.ethz.ch/wiki/JupyterHub

To set up a connection to the cluster with ssh keys, 
follow https://scicomp.ethz.ch/wiki/Accessing_the_clusters#SSH_keys

For convenience, set up the ssh config file on your local mashine as 

.. code::
    Host euler
        HostName euler.ethz.ch
        IdentityFile ~/.ssh/the_file_you_generated_with_ssh_keygen

    Host *
        User your_username
        ForwardAgent yes
        ForwardX11 yes

In order to use git on euler, generate an ssh key in the same way on euler 
and add it to your git profile online. Then, set the ssh config file 
``./ssh/config`` in your home as

.. code::
  Host github.com
      HostName github.com
      IdentityFile ~/.ssh/the_file_you_generated_with_ssh_keygen
    
    Host *
        User your_username

After ``ssh euler`` in your local terminal, you will be on a login node first. 
These are not meant for computations, only for code editing and 
starting computations as 'jobs' from there. For debugging, you can start 
an interactive job and then debug right within the terminal, 
https://scicomp.ethz.ch/wiki/Using_the_batch_system#Interactive_jobs

Once your code is debugged and you would like to run it in larger form,
compile a job script specifying your time and memory requirements. This ensures
that you remember how you ran each computation:
https://scicomp.ethz.ch/wiki/Using_the_batch_system#Job_scripts

You can use your local installation of visual code to connect with the remote extension and use it for code editing. 
However, terminals in vsc work just like normal ones, so follow the best practices above.
