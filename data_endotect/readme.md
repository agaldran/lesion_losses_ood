Hello, glad you are here!

* 1000 train images and masks are downloaded and pre-processed when calling `sh get_endo_data.sh` in the root folder.
* 200 test images and masks corresponding to polyps are the test set of Endotect challenge, which was not part of the Kvasir dataset. 
I already downloaded them from [here](https://endotect.com/) and pre-processed them (resize) for you.
* 20 test images corresponding to images that do not contain polyps are from the ulcerative-colitis images in they hyper-kvasir dataset.
I already downloaded them from [here](https://datasets.simula.no/hyper-kvasir/) and preprocessed them (resize and create black segmentation masks) 

By using the kvasir datasets you are implicitly agreeing to their license terms, which you can check here: https://datasets.simula.no/hyper-kvasir/
You should also probably cite them, https://doi.org/10.1038/s41597-020-00622-y, they are very nice people.

