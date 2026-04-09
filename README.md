### Dependencies and operating systems

python(version=3.8); pytorch(version=1.11.0); CUDA(version=11.3); numpy(version=1.22.4); 

pandas(version=2.0.3); pyarrow(version=15.0.2); sklearn(version=1.3.2).

Ubuntu 20.04 system, Intel(R) Xeon(R) Platinum 8352V CPU @ 2.10GHz, NVIDIA RTX 4090 GPU. 

### Prediction

1.TrsbicHLAI_BA model prediction

（1）Navigate to the 'binding_model_predict' directory. First, modify all file paths in 'TrsbicHLAI_BA.py' to your own file           paths, then run the following command to perform prediction. 
    In the "test" folder, we provide an example test data file: ba_demo_input.csv serves as the test input file. You can refer to     its format to prepare your own input file.
    Please replace '/path/to/your/' with the actual absolute path on your system.

    python TrsbicHLAI_BA.py --input /path/to/your/ba_demo_input.csv --output /path/to/your/ba_demo_output.csv

-- Input: The first column is the peptide sequence, the second column is the HLA name, and the third column is the HLA pseudo-sequence. The HLA pseudo-sequences can be obtained from the MHC_pseudo.dat file in the hla_sequence folder.
-- Output: The path where the prediction results will be saved.


2.TrsbicHLAI_EL model prediction

（1）Navigate to the 'eluted_ligand_model_predict' directory. First, modify all file paths in 'TrsbicHLAI_EL.py' to your own file     paths, then run the following command to perform prediction. 
    In the "test" folder, we provide an example test data file: el_demo_input.csv serves as the test input file. You can refer to     its format to prepare your own input file. 
    Please replace '/path/to/your/' with the actual absolute path on your system.

    python TrsbicHLAI_EL.py --input /path/to/your/el_demo_input.csv --output /path/to/your/el_demo_output.csv

-- Input: The first column is the peptide sequence, the second column is the HLA name, and the third column is the HLA pseudo-sequence. The HLA pseudo-sequences can be obtained from the MHC_pseudo.dat file in the hla_sequence folder.
-- Output: The path where the prediction results will be saved.


3.TrsbicHLAI_TLIM model prediction

（1）Navigate to the 'immunogenicity_model_predict' directory. First, modify all file paths in 'TrsbicHLAI_TLIM.py' to your own       file paths, then run the following command to perform prediction. 
    In the "test" folder, we provide an example test data file: im_demo_input.csv serves as the test input file. You can refer to     its format to prepare your own input file. 
    Please replace '/path/to/your/' with the actual absolute path on your system.

    python TrsbicHLAI_TLIM.py --input /path/to/your/im_demo_input.csv --output /path/to/your/im_demo_output.csv

-- Input: The first column is the peptide sequence, the second column is the HLA name, and the third column is the HLA pseudo-sequence. The HLA pseudo-sequences can be obtained from the MHC_pseudo.dat file in the hla_sequence folder.
-- Output: The path where the prediction results will be saved.

### If you want to reproduce the training process of our model, please follow the procedure below.

1. Train the TrsbicHLAI_BA model
   
   （1）Navigate to the 'binding_model_train' folder, where 'trainval_ba_fold10' is the training data. Please first modify all            file paths in 'binding_train.py' to your own file paths, then execute the following command:

       python binding_train.py --fold 0
       python binding_train.py --fold 1
       python binding_train.py --fold 2
       python binding_train.py --fold 3
       python binding_train.py --fold 4
       python binding_train.py --fold 5
       python binding_train.py --fold 6
       python binding_train.py --fold 7
       python binding_train.py --fold 8
       python binding_train.py --fold 9
    
2. Train the TrsbicHLAI_EL model

   （1）Navigate to the 'eluted_ligand_model_train' folder, where 'trainval_el_fold10' is the training data. Please first modify          all file paths in 'presentation_train.py' to your own file paths, then execute the following command:

       python presentation_train.py --fold 0
       python presentation_train.py --fold 1
       python presentation_train.py --fold 2
       python presentation_train.py --fold 3
       python presentation_train.py --fold 4
       python presentation_train.py --fold 5
       python presentation_train.py --fold 6
       python presentation_train.py --fold 7
       python presentation_train.py --fold 8
       python presentation_train.py --fold 9
       
3. Train the TrsbicHLAI_TLIM model

   （1）Navigate to the 'immunogenicity_model_train' folder, where 'trainval_im_fold10' is the training data. Please first modify        all file paths in 'immunogenicity_train.py' to your own file paths, then execute the following command:

       python immunogenicity_train.py --fold 0
       python immunogenicity_train.py --fold 1
       python immunogenicity_train.py --fold 2
       python immunogenicity_train.py --fold 3
       python immunogenicity_train.py --fold 4
       python immunogenicity_train.py --fold 5
       python immunogenicity_train.py --fold 6
       python immunogenicity_train.py --fold 7
       python immunogenicity_train.py --fold 8
       python immunogenicity_train.py --fold 9
       
       
             