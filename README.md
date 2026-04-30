# LW4_Improving-CNN-Performance

https://colab.research.google.com/drive/19012A2u5flwE4sHLtK6rdKpq_J-q-erz

PART 4: Compare Results (Before vs After) 

Metric                          | Baseline (LW3) | Improved (LW4) | Change
--------------------------------|----------------|----------------|--------
ACCURACY & LOSS                 |                |                |
Training Accuracy               | 95.55%         | 99.54%         | +3.99%
Training Loss                   | 0.1485         | 0.0171         | Lower
Validation Accuracy             | 95.54%         | 99.91%         | +4.37%
Validation Loss                 | 0.1333         | 0.0665         | Lower
Generalization Gap (Train−Val)  | 0.01%          | ~0.00%         | ✓ ≤5%
                                |                |                |
CLASSIFICATION METRICS          |                |                |
Precision (macro avg)           | 96.00%         | 99.76%         | +3.76%
Recall (macro avg)              | 96.00%         | 99.71%         | +3.71%
F1-score (macro avg)            | 96.00%         | 99.73%         | +3.73%
                                |                |                |
OVERALL DISCRIMINATION          |                |                |
AUC Score (OvR)                 | 98.49%         | 99.22%         | +0.73%

GUIDE QUESTIONS (Student Explanation & Reflection)

A. Model Evaluation Analysis 

1. What were the weakest-performing classes based on the confusion matrix?
   - According to your notebook, the dataset is an image dataset loaded from /content/drive/MyDrive/ImageDataset, with class names generated from the folder structure (the Filipino medicinal plant Tsaang Gubat is one confirmed class). By displaying which rows have the lowest diagonal values (accurate predictions) in relation to their row totals, the confusion matrix created in Part 4 of Activity 1 indicates the weakest classes. The most misclassified classes will have the highest off-diagonal values and are visually close to one another (e.g., various medicinal plants with similar leaf forms or hues). To determine which class names caused the most confusion with nearby classes, you must run the notebook.

2. How did Precision, Recall, and F1-score vary across classes?
   - Precision, Recall, and F1 for each class are printed by the classification_report in Part 3 of Activity 1. Variation arises in a dataset of medicinal plants because:
Precision and recall will be high for classes with visually distinctive traits, such as distinctive leaf form, color, and texture.
Classes with similar green leaves or shapes will receive lower scores.
This variation is clearly shown in the bar chart in Part 8 of Activity 1; the classes with shorter bars are the weaker ones.

3. What does a low recall indicate in your model?
   - Low recall for a class (such as "Tsaang Gubat") in your notebook indicates that the model is predicting a large number of its real images as a different plant class, or high False Negatives. After scanning the leaf, the program predicts another medicinal plant rather than Tsaang Gubat. This could be hazardous if the user is attempting to distinguish between deadly and safe herbs in a real-world plant identification application because it would overlook accurate plant identifications.
     
4. How does AUC score reflect model performance compared to accuracy?
   - In Part 7, your notebook uses roc_auc_score(... multi_class='ovr') to calculate the overall AUC, and in Part 6, it draws ROC curves for each class. For your classifier for medicinal plants:
The percentage of all predictions that were accurate overall is called accuracy.
AUC indicates how well the model distinguishes "this class vs. all others" for each plant class across all potential thresholds, not just the standard 0.5 cutoff.
Accuracy may appear good if one plant class predominates in your sample (class imbalance), but AUC indicates that the model has trouble with minority classes. Strong per-class discrimination is confirmed by AUC values near 1.0 per class in the legend of Part 6.

B. Model Improvement 

5. How did data augmentation affect validation accuracy?
   - RandomFlip ("horizontal_and_vertical"), RandomRotation(0.2), RandomZoom(0.2), and RandomContrast(0.2) are applied to the training pipeline in Activity 3 by Enhancement 1. This is particularly crucial for a dataset of medicinal plants because:
Photographs of plants are captured in the real world under various lighting and angle situations.
The model learns through augmentation that a 45° rotated Tsaang Gubat leaf is still Tsaang Gubat.
This directly increases validation accuracy in history.history['val_accuracy'] by decreasing overfitting on the initial training photos.

6. Why is Batch Normalization important in CNNs?
   - BatchNormalization() is added to your enhanced model in Enhancement 2 following each Conv2D layer (after the 32-filter, 64-filter, and 128-filter conv layers). In particular, this stabilizes the distribution of pixel-level feature activations between layers for your plant classifier, preventing crazily scaled inputs from reaching the deeper layers. It functions as a regularizer, which lessens the need for extremely large Dropout values, and accelerates convergence (you'll notice the training loss drop faster per epoch in history.history['loss]).
     
7. What role did Dropout play in improving your model?
   - Enhancement 2 employs Dropout (0.5) prior to the output Dense layer and Dropout (0.4) following the final MaxPooling. In the absence of Dropout, the baseline model probably committed particular training photos of every plant to memory, resulting in high training accuracy but low validation accuracy. Dropout prevents neurons from co-adapting, therefore the model has to build robust, dispersed representations of the distinctive appearance of a Tsaang Gubat leaf. The accuracy plot in Part 5 should show a noticeable reduction in the generalization gap (training accuracy − validation accuracy).
     
8. How did Early Stopping prevent overfitting? 
  - EarlyStopping (monitor='val_loss', patience=3, restore_best_weights=True) is configured in Enhancement 4. Enhancement 5 allows training to run for up to 20 epochs; however, training automatically stops and the optimal weights are restored if val_loss stops improving for three consecutive epochs. Instead of continuing to train while the model gradually memorizes the training plant photos, you should notice the training halt before epoch 20, at the moment where validation loss was at its lowest, on the accuracy/loss plots from Part 5.
    
C. Performance Comparison 

9. What improvements were observed after modifying the model? 
  - Higher validation accuracy, decreased validation loss, improved per-class F1 scores (particularly for previously weak plant classes), and a higher overall AUC should be observed when comparing the classification_report and AUC score from Activity 1 (baseline) to those rerun in Activity 3 Part 3 (enhanced). There should be fewer off-diagonal misclassifications between plant species that look similar in the confusion matrix.
    
10. Which enhancement contributed the most to performance improvement?Why?
  - Data Augmentation (Enhancement 1) and Dropout (Enhancement 2) made the biggest contributions to a dataset of images of medicinal plants. By training the model on a variety of transformations, augmentation directly addresses the very varying orientation, illumination, and zoom level of real-world plant photos. Dropout keeps the model from depending too much on pixel patterns unique to a given dataset. While augmentation and dropout target the fundamental issue of overfitting on a small-to-medium plant dataset, batch normalization and early stopping are crucial support strategies.
    
11. Did the gap between training and validation accuracy decrease? Explain.
  - Yes. The two lines in the Part 5 accuracy plot (history.history['accuracy'] vs. history.history['val_accuracy']) should be substantially closer together in the upgraded model than in the baseline. The baseline most likely displayed a significant disparity, with validation accuracy plateauing or perhaps declining as training accuracy shot toward 99%. By regularizing the training process with augmentation, BatchNorm, and Dropout, the improved model achieves the lab's target generalization gap of ≤5%.
    
D. Explainability (Grad-CAM Integration) 

12. How did Grad-CAM help in understanding model predictions? 
  - In Activity 2, your notebook creates a Grad-CAM function that, by beginning with rescaling_1 and identifying conv2d_5 as the final convolutional layer, expressly avoids the augmentation layer (sequential_1). The pixels of your Tsaang Gubat test picture (/content/drive/MyDrive/ImageDataset/TSAANG GUBAT/test.jpg) that most strongly activated the prediction are displayed in the heatmap created in Part 5 and the overlay in Part 6. As a result, you can observe whether the CNN is concentrating on the leaf structure itself or on unimportant background components, turning the model from a black box into an understandable system.
        
13. Did the improved model focus on more relevant regions? Provide evidence. 14. Why is explainability important in real-world AI applications?
 - After retraining with Enhancement 2's deeper architecture (32→64→128 filters with BatchNorm), the improved model learns hierarchically richer features. Running Grad-CAM on the same test.jpg Tsaang Gubat image before and after should show a tighter, more leaf-centered heatmap in the improved model. If the baseline heatmap highlighted soil/background around the plant and the improved heatmap highlights the leaf veins, edges, or texture — that is direct visual evidence that the enhancements made the model learn botanically meaningful features.

14. Why is explainability important in real-world AI applications?
 - Explainability is crucial for a medicinal plant classifier such as yours. A farmer or healthcare professional must have faith in the forecast if they use this app to identify a plant before consuming or prescribing it. Grad-CAM enables domain specialists (pharmacists, botanists) to confirm that the model accurately identified the plant for the correct reasons, independent of background artifacts, image watermarks, or lighting conditions. Additionally, it aids developers in debugging failure instances. For example, if the heatmap indicates that the model is activating on soil instead of leaves, it indicates that improved data collection or preparation is required.
