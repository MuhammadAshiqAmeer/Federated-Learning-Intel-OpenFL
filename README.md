# Federated lemon-melon CNN classifier(Intel-open-fl)
### Director based federated learning workflow of a custom Lemon-melon CNN classifier done in Intel-Open-FL.
#### Aim to approach and customize Intel-openfl by building an FL model from a custom dataset and an ML experiment defined, using some non default aggregation function.

#### Overview of the repo:

* Contains two directories, director and envoy.
* Director folder has a .ipynb (notebook) file where the experiment is setup by a Data scientist, which contains code for connecting to federation devices such as aggregators and collaborators, setting up federation tasks for different connected devices, setting federation algorithm etc.
* It also contains director config(.yaml) file which sets the configuration of listen host and port of director, also sample and taget shape of data united accross the federation.
* Envoy folder contains the envoy config (.yaml) file that sets local shard descriptor, rank of the collaborator and worldsize etc., also a shard descreptor(.py) file that is responsible for splitting a single dataset according to collaborator's rank so that every collaborator will get different data during the simulation.(Not the case in real world scenario, there will be independent data associated with each collaborators,so no need of shard_desc)

#### Steps to start a federation:
If you are cloning this repo then the steps are:

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
    - The jupyter notebook in the director folder contains the detailed code to run the federated learning experiment.

Steps to implement an FL experiment from scratch:

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
      
  - on Director:
  
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
  - For director based approach, the initial connection establishment between Director and Envoys (both running on docker) can be done by exposing a port of director, envoys connected to that port(director listening ip set as 0.0.0.0 to accept all incoming connections through the exposed port and envoys are given with the FQDN/IP of director machine with exposed port No. of director). 
  - But when we start the experiment, director starts an aggregator service with new port and ip which will be not exposed at the time of docker image creation. So the collaborators started by the envoys will fail to connect to the aggreagator service.
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
  - Go to the mounted docker directory in envoy machine.
  - Start the envoys in envoy machines
  ```
  bash StartEnvoy
  ```

### Reference:
  - [Official intel open federated learning documentation](https://openfl.readthedocs.io/en/latest/index.html)
