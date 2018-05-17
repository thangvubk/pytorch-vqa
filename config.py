# paths
qa_path = 'data/annotations'  # directory containing the question and annotation jsons
train_path = 'data/images/train2014'  # directory of training images
val_path = 'data/images/val2014'  # directory of validation images
test_path = 'data/images/test2015'  # directory of test images
preprocessed_path = 'data/resnet-14x14.h5'  # path where preprocessed features are saved to and loaded from
vocabulary_path = 'data/vocab3000.json'  # path where the used vocabularies for question and answers are saved to

task = 'OpenEnded'
dataset = 'mscoco'

# preprocess config
preprocess_batch_size = 64
image_size = 448  # scale shorter end of image to this size and centre crop
output_size = image_size // 32  # size of the feature maps after processing through a network
output_features = 2048 # (512 for VGG and 2048 for ResNet) number of feature maps thereof
central_fraction = 0.875  # only take this much of the centre when scaling and centre cropping
max_answers = 3000
