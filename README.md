# Species-Flow-Networks

## Steps for generating SF-HON from the raw ship movements and port data.

Run 00_sort_clean_moves.ipynb to clean and sort the raw ship movements based on ship ID and arrival and sail time.

Run 01_create_traces.ipynb to build traces from cleaned ship movements.

Run 02_Prepare_HON_Input.ipynb to claculate path probablities for ballast and biofouling transfer

Run 03_Build-SF-HON.ipynb to build Species-Flow Higher-Order Network (SF-HON) using the probabilities and sequences from the previous step

Run 04_make_HONet_dict.ipynb to build the phyiscal network representation, and prepare the network for clustering.

