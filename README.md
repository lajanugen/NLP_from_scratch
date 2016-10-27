# NLP_from_scratch

* Data

- Make sure the treebank data is in '/z/data/treebank_text/'. 
- For NER and Chunking, the datasets are available in this repository.

* Tagsets

- For NER and Chunking, the IOB tagging scheme is used. Note that this is different from the IOBES tagging scheme used in the paper.

Example command lines: 
* th main.lua -task POS -mode train -gpu 4 -desc "POS experiment"

- Specifies the task to be POS (Options: POS/NER/Chunking)

- Training mode

- GPU 4 is to be used

- The description of the experiment is "POS experiment". This description will be appended as a new line to 'results/descriptions' with a number at the beginning. A folder will be created in 'results' with this number as its name and all the training files, logs will be stored in this directory.
To reuse the previous directory, specify '-desc 0'

* th main.lua -task POS -mode test -gpu 4

- Evaluates a previously trained model on the test set for the task.
- Make sure the pre-trained model is specified in pre_init_mdl_path and pre_init_mdl_name in main.lua (line numbers 62, 63) which respectively indicate the folder in which the trained model is stored (in results) and the model id.

* th main.lua -task POS -mode demo -gpu 4 -input "This is a great course ."

- Runs the pre-trained model on an input sentence. As for the test mode, make sure that the pre-trained model is specified in main.lua

* Modify other parameters in main.lua for training as necessary

* The demo version is currently supported only for the Word Level Likelihood objective. Demo support for the Sentence Level Likelihood objective will be added soon.
