# Federated_lemon-melon_CNN_classifier(Intel-open-fl)
### Director based federated learning workflow of a custom Lemon-melon CNN classifier done in Intel-Open-FL.

* Contains two directories, director and envoy.
* Director folder has a .ipynb (notebook) file where the experiment is setup by a Data scientist, which contains code for connecting to federation devices such as aggregators and collaborators, setting up federation tasks for diffrent connected devices, setting federation algorithm etc.
* It also contains director config(.yaml) file which sets the configuration of listen host and port of director, also sample and taget shape of data united accross the federation.
* Envoy folder contains the envoy config (.yaml) file that sets local shard descriptor, rank of the collaborator and worldsize etc., also a shard descreptor(.py) file that is responsible for splitting a single dataset according to collaborator's rank so that every collaborator will get different data during the simulation.(Not the case in real world scenario, there will be independent data associated with each collaborators,so no need of shard_desc)

#### Steps to start a federation:
If you are cloning this repo then the steps are:

* Start the director:
  - On director device, go inside director folder in the terminal.
  - set up the listen_host to your FQDN or IP and available port number.
  - If mTLS protection is not set up, run this command:(easy way)
  ```
      fx director start --disable-tls -c director_config.yaml
  ```
- If you have a federation with PKI certificates, run this command:
  ```
      fx director start -c director_config.yaml \
     -rc cert/root_ca.crt \
     -pk cert/priv.key \
     -oc cert/open.crt
   ```
  - You can see the log info of director name,port etc. when it starts

* Start the envoy:
  - On envoy device, go inside envoy folder in terminal. 
  - Set up the sample and target shape(if your data is different)
  - If mTLS protection is not set up, run this command:
    ```
      fx envoy start \
          -n "ENVOY_NAME" \
          --disable-tls \
          --envoy-config-path envoy_config.yaml \
          -dh director_fqdn \
          -dp port
    ```
  - If you have a federation with PKI certificates, run this command:
    ```
      fx envoy start \
      -n "ENVOY_NAME" \
      --envoy-config-path envoy_config.yaml \
      -dh director_fqdn \
      -dp port \
      -rc cert/root_ca.crt \
      -pk cert/"ENVOY_NAME".key \
      -oc cert/"ENVOY_NAME".crt
    ```
   - You can see the experiment recieved and data loaded status in the log info when it starts.
  
* Setup an experiment
    - The process of defining an experiment is decoupled from the process of establishing a federation. The Experiment manager (or data scientist) is able to prepare an experiment in a Python environment. Then the Experiment manager registers experiments into the federation using Interactive Python API (Beta) that is allow to communicate with the Director using a gRPC client.
    - The Open Federated Learning (OpenFL) interactive Python API enables the Experiment manager (data scientists) to define and start a federated learning experiment from a single entry point: a Jupyter* notebook or a Python script.
    - The jupyter notebook in the director folder contains the detailed code to run the federated learning experiment.

Steps to implement a FL experiment from scratch:

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

### Reference:
  - [Official intel open federated learning documentation](https://openfl.readthedocs.io/en/latest/index.html)
