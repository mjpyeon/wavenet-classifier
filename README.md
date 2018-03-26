# wavenet-classifier
Modified WaveNet Architecture for Supervised Learning Tasks

The goal of this project is to provide a model for speech classification and/or regression using WaveNet architecture which is originally designed as a generative model.
So, this project provides an API for various supervised learning tasks related to speechs.
Note that our implementation is based on keras2 on tensorflow background.

##Usage
<pre>
<code>
from WaveNetClassifier import WaveNetClassifier

wnc = WaveNetClassifier((96000,), (10,), kernel_size = 2, dilation_depth = 9, n_filters = 40, task = 'classification')
wnc.fit(X_train, y_train, validation_data = (X_val, y_val), epochs = 100, batch_size = 32, optimizer='adam', save=True, save_dir='./')
y_pred = wnc.predict(X_test)
</code>
</pre>
