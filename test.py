import numpy as np
from skimage.metrics import hausdorff_distance
#Test of 1 slice
def dice_coefficient(y_true, y_pred):
    if np.sum(y_true) + np.sum(y_pred) == 0:
        return 0
    intersection = np.sum(y_true * y_pred)
    return 2 * intersection / (np.sum(y_true) + np.sum(y_pred))

# def hausdorff_distance(y_true, y_pred):
#     points_true = np.column_stack(np.nonzero(y_true))
#     points_pred = np.column_stack(np.nonzero(y_pred))
#     return max(directed_hausdorff(points_true, points_pred)[0], directed_hausdorff(points_pred, points_true)[0])

def metrics_calculation(y_true, y_pred):
    results = []
    results1=[]
    for i in range(1, 5):
        temp1 = np.zeros(y_true.shape, dtype='int')
        temp2 = np.zeros(y_true.shape, dtype='int')
        mask1 = (y_true == i)
        mask2 = (y_pred == i)
        temp1[mask1] = 1
        temp2[mask2] = 1
        results.append([dice_coefficient(temp1, temp2), hausdorff_distance(temp2, temp1)])
        #results1.append(metrics_calculation_sitk(temp1,temp2))
    return results


y_true = np.array([[1, 1, 0, 0, 1, 0, 1, 1, 0],[1, 1, 0, 0, 1, 0, 1, 1, 0]])
y_pred = np.array([[1, 0, 0, 1, 1, 0, 1, 0, 0], [1, 1, 0, 0, 1, 0, 1, 1, 0]])


#Using the lib mentioned in the competition
import SimpleITK as sitk
def metrics_calculation_sitk(y_true:np.ndarray, y_pred:np.ndarray):
    image1 = sitk.GetImageFromArray(y_true)
    image2 = sitk.GetImageFromArray(y_pred)
    overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
    overlap_measures_filter.Execute(image1, image2)
    hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()
    hausdorff_distance_filter.Execute(image1, image2)
    dice_coefficient1 = overlap_measures_filter.GetDiceCoefficient()
    hausdorff_distance1 = hausdorff_distance_filter.GetHausdorffDistance()

    return [dice_coefficient1, hausdorff_distance1]


re = metrics_calculation(y_true, y_pred)
print(re)

re1 = metrics_calculation_sitk(y_true, y_pred)
print(re1)

