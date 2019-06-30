# Flower-Classification
Classifies 5 types of flower using concept of transfer learning

I've trained this model on Ubuntu 18.04 inside a python virtualenv.
To install virtualenv in Ubuntu 18.04.

> sudo apt install python3-pip<br/>
pip3 install virtualenv<br/>
virtualenv flowerdetection<br/>
cd flowerdetection<br/>

Before you start any training, you'll need a set of images to teach the network about the new classes you want to recognize. There's a later section that explains how to prepare your own images, but to make it easy we've created an archive of creative-commons licensed flower photos to use initially. To get the set of flower photos, run these commands:

> curl -LO http://download.tensorflow.org/example_images/flower_photos.tgz
> tar xzf flower_photos.tgz

# Configure your MobileNet

In this exercise, we will retrain a MobileNet. MobileNet is a a small efficient convolutional neural network. "Convolutional" just means that the same calculations are performed at each location in the image.

The MobileNet is configurable in two ways:

    Input image resolution: 128,160,192, or 224px. Unsurprisingly, feeding in a higher resolution image takes more processing time, but results in better classification accuracy.
    The relative size of the model as a fraction of the largest MobileNet: 1.0, 0.75, 0.50, or 0.25.

We will use 224 0.5 for this codelab.

With the recommended settings, it typically takes only a couple of minutes to retrain on a laptop. You will pass the settings inside Linux shell variables. Set those variables in your shell: 

> IMAGE_SIZE=224
ARCHITECTURE="mobilenet_0.50_${IMAGE_SIZE}"

# Start TensorBoard

Before starting the training, launch tensorboard in the background. TensorBoard is a monitoring and inspection tool included with tensorflow. You will use it to monitor the training progress.

> tensorboard --logdir tf_files/training_summaries &

This command will fail with the following error if you already have a tensorboard process running:

ERROR:tensorflow:TensorBoard attempted to bind to port 6006, but it was already in use

You can kill all existing TensorBoard instances with:

> pkill -f "tensorboard"

Download the retraining script from this : https://github.com/tensorflow/hub/blob/master/examples/image_retraining/retrain.py
Then run this command:

> python -m scripts.retrain \
  --bottleneck_dir=tf_files/bottlenecks \
  --how_many_training_steps=500 \
  --model_dir=tf_files/models/ \
  --summaries_dir=tf_files/training_summaries/"${ARCHITECTURE}" \
  --output_graph=tf_files/retrained_graph.pb \
  --output_labels=tf_files/retrained_labels.txt \
  --architecture="${ARCHITECTURE}" \
  --image_dir=tf_files/flower_photos

Note that this step will take a while.

The first retraining command iterates only 500 times. You can very likely get improved results (i.e. higher accuracy) by training for longer. To get this improvement, remove the parameter --how_many_training_steps to use the default 4,000 iterations.

The first retraining command iterates only 500 times. You can very likely get improved results (i.e. higher accuracy) by training for longer. To get this improvement, remove the parameter --how_many_training_steps to use the default 4,000 iterations.

# Using the Retrained Model

The retraining script writes data to the following two files:

    tf_files/retrained_graph.pb, which contains a version of the selected network with a final layer retrained on your categories.
    tf_files/retrained_labels.txt, which is a text file containing labels.


# Classifying an image

To classify images, download this script: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/label_image/label_image.py

> python -m scripts.label_image \
    --graph=tf_files/retrained_graph.pb  \
    --image=tf_files/flower_photos/daisy/21652746_cc379e0eea_m.jpg
    
You might get output like:

daisy (score = 0.99071)
sunflowers (score = 0.00595)
dandelion (score = 0.00252)
roses (score = 0.00049)
tulips (score = 0.00032)

# Trying Other Hyperparameters (Optional)

The retraining script has several other command line options you can use.

You can read about these options in the help for the retrain script:

> python -m scripts.retrain -h

Try adjusting some of these options to see if you can increase the final validation accuracy.

For example, the --learning_rate parameter controls the magnitude of the updates to the final layer during training. So far we have left it out, so the program has used the default learning_rate value of 0.01. If you specify a small learning_rate, like 0.005, the training will take longer, but the overall precision might increase. Higher values of learning_rate, like 1.0, could train faster, but typically reduces precision, or even makes training unstable.

You need to experiment carefully to see what works for your case.
