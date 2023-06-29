####  Note  :black_nib:  : Switch branches to see different experiments.
# Federated MNIST CNN classifier(Intel-open-fl)
#### Director based federated learning workflow of a custom MNIST CNN classifier done in Intel-Open-FL. Aim to approach and customize Intel-openfl by building an FL model from a custom dataset and an ML experiment defined, using some non default aggregation function.

## Repository Overview :
This repository provides an detailed steps and code of Federated Learning experiment using OpenFL. The repository consists of the following components:
* The "Director" folder contains a configuration file named director_config.yaml, which sets the listening host, port, and other configuration parameters for the director. It also includes a sample and target shape definition for the unified data across the federation.
* Inside the "Envoy" folder, you will find the envoy_config.yaml file, which defines the local shard descriptor, collaborator rank, world size, and other settings for the envoy. Additionally, there is a shard_descriptor.py file responsible for loading data from local collaborators to initiate the experiment.
* The "Workspace" folder contains a Jupyter Notebook (.ipynb) file where a specific federated learning experiment is setup. The notebook includes code to connect to federation devices, creating machine learning model to train, configure federation tasks, choose the federation algorithm, etc.
* The "Patches" folder shows the modified stream metrics function, to record experiment metric values along with streaming during runtime.
* Bash scripts:
    - Docker directory:
      * Deploy - This script creates a Dockerfile, builds an image, and mounts a local directory to share the experiment. It is used for the initial setup and             image creation.
      * Start -  Use this script to start the container if it has already been created.
    - Scripts directory:
        * StartEnvoy - Run this script to start the envoy nodes.
        * StartDirector - Execute this script to start the director service.
        * distribute.sh & split.sh - These scripts are used for dataset splitting and distribution to collaborators during simulations.
        * maximum.sh - This script records the metric name and maximum value from the logfile after the experiment.



####  Note  :black_nib: 
  * Please ensure to adjust the FQDN/IP and port settings in relevant files such as director.yaml, the experiment notebook, envoy startup commands, and bash scripts, according to your machine's configuration.


## Steps to start a federation:
### Install Intel OpenFL
  - Prerequisites: Python 3.8 (>=3.6, <3.9) virtual environment using Python venv or Anaconda
  -  Activate the created virtual environment.
  - Install OpenFL:
   * Using PyPl, run the following command:
       ```
       python -m pip install openfl
       ```
   * Or from source:
     - Clone the repository:
       ```
       git clone https://github.com/intel/openfl.git
       ```
     - Install build tools before installing OpenFL:
       ```
       python -m pip install -U pip setuptools wheel
       cd openfl/
       python -m pip install .
       ```
Once you have completed the installation steps, you are ready to use OpenFL in your Python environment. Running the 'fx' command confirms the successful installation of OpenFL. 

### To run this FL experiment:
   * Start the director:
      - On director device, go to director folder in the terminal.
      - set up the listen_host to your FQDN or IP and available port number in director.yaml file.
      - If mTLS protection is not set up, run this command:(easy way)
      ```
          fx director start --disable-tls -c director_config.yaml
      ```
      - If you have a federation with PKI certificates, run this command:
      ```
          fx director start -c director_config.yaml -rc cert/root_ca.crt -pk cert/priv.key -oc cert/open.crt
       ```
      - You can see the log info of director name,port etc. when it starts
    
   * Start the envoy:
      - On envoy device, go to envoy folder in terminal.
      - Install packages in requirements.txt file
      ```
      pip install -r requirements.txt
      ```
      - Set up the sample and target shape(if your data is different) and also shard descriptor(.py) file address in envoy_congig.yaml file
      - If mTLS protection is not set up, run this command:
        ```
          fx envoy start -n "ENVOY_NAME" --disable-tls --envoy-config-path envoy_config.yaml -dh director_fqdn -dp port
        ```
      - If you have a federation with PKI certificates, run this command:
        ```
          fx envoy start -n "ENVOY_NAME" --envoy-config-path envoy_config.yaml -dh director_fqdn -dp port -rc cert/root_ca.crt -pk cert/"ENVOY_NAME".key -oc cert/"ENVOY_NAME".crt
        ```
       - You can see the experiment recieved and data loaded status in the log info when it starts.
      
   * Setup an experiment
        - The process of defining an experiment is decoupled from the process of establishing a federation. The Experiment manager (or data scientist) is able to prepare an experiment in a Python environment. Then the Experiment manager registers experiments into the federation using Interactive Python API (Beta) that is allow to communicate with the Director using a gRPC client.
        - The Open Federated Learning (OpenFL) interactive Python API enables the Experiment manager (data scientists) to define and start a federated learning experiment from a single entry point: a Jupyter* notebook or a Python script.
        - The jupyter notebook in the workspace folder contains the detailed code to run the federated learning experiment.
        - On a machine start jupyter server and open the notebook to run it.

#### To build a different custom FL experiment

  - On director :
  
    - Create an FL workspace:
    
      ```
      fx director create-workspace -p path/to/director_workspace_dir
      ```
    - Modify the Director config file according to your federation setup.
    - Start the director by using one of the above mentioned steps.
    
  - On envoy:
     
     - Create a workspace:
       ```
       fx envoy create-workspace -p path/to/envoy_workspace_dir
       ```
      - Modify the Envoy config file and local shard descriptor template.
        - Complete the shard descriptor template field with the address of the local shard descriptor class.
        - Create a shard_desc.py file and code it accordingly([see the documentation](https://openfl.readthedocs.io/en/latest/running_the_federation.html#collaborator-manager-set-up-the-envoy))
      - start the envoy with one of the above mentioned methods.
      
  - on Experiment manager's machine:
  
    - Create a jupyter notebook file.
    - Describe an experiment([see documentation](https://openfl.readthedocs.io/en/latest/running_the_federation.html#experiment-manager-describe-an-experiment))
    
Changing the default weighted average aggregation algorithm:

   - Inside the openfl directory openfl / component / aggregation_functions or inside openfl / interface / aggegation_function some builtin aggregation functions are given.
    example
       ```
       from openfl.interface.aggregation_functions import Median
       TI = TaskInterface()
       agg_fn = Median()
       @TI.register_fl_task(model='model', data_loader='train_loader', device='device', optimizer='optimizer')
       @TI.set_aggregation_function(agg_fn)
       ```
   - You can code your own aggregation function and import it into the notebook and set aggregation function using TaskInterface decorators.
   - You can also code it in the same notebook in seperate cell and import it and set aggregation function using the task interface decorators.
   - [Overriding aggregation functions](https://openfl.readthedocs.io/en/latest/overriding_agg_fn.html)
   

### Using Docker:
  - Docker can be used to deploy FL experiments in openfl
  - There is an image of openfl in [Dockerhub](https://hub.docker.com/r/intel/openfl)
  - For director based approach, the initial connection establishment between Director and Envoys (both running on docker) can be done by exposing a port of director, making envoys connect to that port(director listening ip set as 0.0.0.0 to accept all incoming connections through the exposed port)and with the FQDN/IP of director machine(not 0.0.0.0) with exposed port No. of director. 
  - But when we start the experiment, director starts an aggregator service with new port and ip which will be not exposed at the time of docker image creation. As the workspace were already exported to the envoys at the starting of experiment(which containes the aggregator ip and port created by director to connect collaborators with), the collaborators started by the envoys will fail to connect to the aggreagator service.
  - Solution is to start the director without docker(since it will be a static single machine) and deploy envoys using docker(for easy distribution).
  - The bash scripts are included here.
  
  
  - Deploy a docker container with openfl image including packeges needed for the experiment to run(in envoy machines):
  ```
  bash Deploy
  ```
  - If already created container, Start the container.
  ```
  bash Start
  ```
  - Start the director in director machine(without docker)
  - Go to the mounted docker directory (lm_fl_experiment)in envoy machine.
  - Start the envoys in envoy machines
  ```
  bash StartEnvoy
  ```

### Reference:
  - [Official intel open federated learning documentation](https://openfl.readthedocs.io/en/latest/index.html)
