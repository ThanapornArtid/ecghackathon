# ecghackathon
1. requires libraries:
   **pip install torch torchvision pandas Pillow scikit-learn torcheval tensorboard**
2. Run the **trainer.py** script to begin the training process. This will save the best model weights based on validation loss as model_best_vloss.pth.
3. run **tensorboard --logdir=runs** to monitor the Loss, Accuracy, F1-Score
4. you can create more model at model.py and test by change the import **from model import xxx**
