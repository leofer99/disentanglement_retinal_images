# Disentanglement and Assessment of Shortcuts in Ophthalmological Retinal Imaging Exams
## About
This repository contains the implementation code for the [paper](https://arxiv.org/abs/2507.09640) "Disentanglement and Assessment of Shortcuts in Ophthalmological Retinal Imaging Exams" by Leonor Fernandes, Tiago Gonçalves, João Matos, Luis Filipe Nakayama and Jaime S. Cardoso.

## Contributors
* Leonor Fernandes ([GitHub](https://github.com/leofer99) / Hugging Face)
* Tiago Gonçalves ([GitHub](https://github.com/TiagoFilipeSousaGoncalves) / [Hugging Face](https://huggingface.co/tiagofilipesousagoncalves))
* João Matos ([GitHub](https://github.com/joamats) / [Hugging Face](https://huggingface.co/joamats))
* Luis Filipe Nakayama (GitHub / Hugging Face)
* Jaime S. Cardoso (GitHub / Hugging Face)

## Abstract
Diabetic retinopathy (DR) is a leading cause of vision loss in working-age adults. While screening reduces the risk of blindness, traditional imaging is often costly and inaccessible. Artificial intelligence (AI) algorithms present a scalable diagnostic solution, but concerns regarding fairness and generalization persist. This work evaluates the fairness and performance of image-trained models in DR prediction, as well as the impact of disentanglement as a bias mitigation technique, using the diverse mBRSET fundus dataset. Three models, ConvNeXt V2, DINOv2, and Swin V2, were trained on macula images to predict DR and sensitive attributes (SAs) (e.g., age and gender/sex). Fairness was assessed between subgroups of SAs, and disentanglement was applied to reduce bias. All models achieved high DR prediction performance in diagnosing (up to 94% AUROC) and could reasonably predict age and gender/sex (91% and 77% AUROC, respectively). Fairness assessment suggests disparities, such as a 10% AUROC gap between age groups in DINOv2. Disentangling SAs from DR prediction had varying results, depending on the model selected. Disentanglement improved DINOv2 performance (2% AUROC gain), but led to performance drops in ConvNeXt V2 and Swin V2 (7% and 3%, respectively). These findings highlight the complexity of disentangling fine-grained features in fundus imaging and emphasize the importance of fairness in medical imaging AI to ensure equitable and reliable healthcare solutions.

Please, refer to the [original Hugging Face repository](https://huggingface.co/tiagofilipesousagoncalves/disentanglement_retinal_images) for the models' weights.