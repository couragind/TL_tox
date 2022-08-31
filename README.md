# TL_tox

This code contains three folders. The baseline_model only trained on tox21 dataset, which has 12 endpoints, and is uesed as baseline. The initial_model is only trained on CP dataset, and this is a multitask model. The TL_model is trained on tox21 dataset again based on the initial_model, this is a transfer learning model.

Dont forget to replace with your own data path. They are in ...stra_vgg19.py, ...batch_vgg19.py, tox_GNN_image_multitask_nomask_1696.py
