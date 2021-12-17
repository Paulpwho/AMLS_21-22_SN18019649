# AMLS_21-22_SN18019649

Instructions on how to compile and use your code should be created in the repository
* Introduction
    * The deliverables of the assignment were to develop machine learning models for binary (Task A) and multiclass (Task B) classification.
    * Prelimiary experiments were carried out to test the feature extraction stage. The classifier used was logistic regression.
    * For Task A, two classifers were used: Random Forest and K-means clustering.
    * For Task B, an SVM was used.
* A brief description of the organization of the files.
    * All files are in the my_app folder
    * The dataset and csv labels file are in the datasets folder
    * All the models are placed in the my_app directory.
* The role of each file.
    * test1.py - Threshold + Logistic regression
    * test2.py - PCA feature extraction testing
    * test3.py - Histogram feature extraction
    * test4.py - Inital Random forests testing
    * test5.py - Intial testing for Task B (logistic regression)
    * test6.py - PCA investigation: varying the number of components
    * test7.py - Hyperparameter tuning Random Forest – Random search
    * test8.py – Hyperparameter tuning Random Forest – Grid search
    * test9.py – Out of bag error of Random Forest
    * test10.py – Unsupervised clustering methods – K means
    * test11.py – K means manual hyperparameter tuning
    * test12.py – K means automatic hyperparameter tuning
    * test13.py – SVM inital model
    * test14.py – SVM with manual hyperparameter tuning
    * test15.py – Comparting SVMs with different kernel functions
    * test16.py – Full Automated hyperparameter tuning – Random search
* How to run your code.
    * After installing all packages from requirement.txt, all the python programs should be able to run independently. There is no need to run setup.py
* Necessary packages or header files (N.B. All necessary packages are also listed in the requirements.txt file):
    * datetime
    * numpy
    * scipy
    * matplotlib
    * numpy
    * pandas
    * pandas
    * h5py
    * sklearn
    * tensorflow
    * keras
    * torch
    * absl-py==0.15.0
    * astunparse==1.6.3
    * attrs==21.2.0
    * backcall==0.2.0
    * black==21.12b0
    * cached-property==1.5.2
    * cachetools==4.2.4
    * certifi==2021.10.8
    * charset-normalizer==2.0.7
    * clang==5.0
    * click==8.0.3
    * colorama==0.4.4
    * cycler==0.10.0
    * DateTime==4.3
    * decorator==5.1.0
    * evaluation==0.0.2
    * flatbuffers==1.12
    * gast==0.4.0
    * glog==0.3.1
    * google-auth==2.3.0
    * google-auth-oauthlib==0.4.6
    * google-pasta==0.2.0
    * grpcio==1.41.0
    * h5py==3.1.0
    * idna==3.3
    * imageio==2.11.1
    * importlib-metadata==4.8.1
    * importlib-resources==5.4.0
    * ipython==7.30.1
    * ipython-genutils==0.2.0
    * jedi==0.18.1
    * Jinja2==3.0.3
    * joblib==1.1.0
    * jsonschema==4.2.1
    * jupyter-core==4.9.1
    * keras==2.6.0
    * Keras-Preprocessing==1.1.2
    * kiwisolver==1.3.2
    * Markdown==3.3.4
    * MarkupSafe==2.0.1
    * matplotlib==3.4.3
    * matplotlib-inline==0.1.3
    * mistune==2.0.0
    * mypy-extensions==0.4.3
    * nbformat==5.1.3
    * networkx==2.6.3
    * numpy==1.19.5
    * oauthlib==3.1.1
    * opt-einsum==3.3.0
    * pandas==1.3.4
    * parso==0.8.3
    * pathspec==0.9.0
    * pickleshare==0.7.5
    * Pillow==8.4.0
    * platformdirs==2.4.0
    * prompt-toolkit==3.0.24
    * protobuf==3.19.0
    * pyasn1==0.4.8
    * pyasn1-modules==0.2.8
    * Pygments==2.10.0
    * pyparsing==3.0.1
    * pyrsistent==0.18.0
    * python-dateutil==2.8.2
    * python-gflags==3.1.2
    * pytz==2021.3
    * PyWavelets==1.2.0
    * pywin32==302
    * requests==2.26.0
    * requests-oauthlib==1.3.0
    * rsa==4.7.2
    * scikit-image==0.18.3
    * scikit-learn==1.0
    * scipy==1.7.1
    * six==1.15.0
    * sklearn==0.0
    * sklearn-evaluation==0.5.7
    * tabulate==0.8.9
    * tensorboard==2.7.0
    * tensorboard-data-server==0.6.1
    * tensorboard-plugin-wit==1.8.0
    * tensorflow-estimator==2.6.0
    * termcolor==1.1.0
    * threadpoolctl==3.0.0
    * tifffile==2021.11.2
    * tomli==1.2.2
    * torch==1.10.0
    * traitlets==5.1.1
    * typed-ast==1.5.1
    * typeguard==2.13.2
    * typing_extensions==4.0.1
    * urllib3==1.26.7
    * wcwidth==0.2.5
    * Werkzeug==2.0.2
    * wrapt==1.12.1
    * zipp==3.6.0
    * zope.interface==5.4.0
