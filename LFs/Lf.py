import numpy as np
import re
import enum

import sys
sys.path.append('../../')

from spear.labeling import labeling_function, LFSet, ABSTAIN, preprocessor
import torch

classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa','train', 'tvmonitor', 'background']

class ClassLabels(enum.Enum):
    aeroplane : 0
    bicycle : 1
    bird : 2
    boat : 3
    bottle : 4
    bus : 5
    car : 6
    cat : 7
    chair : 8
    cow : 9
    diningtable : 10
    dog : 11
    horse : 12
    motorbike : 13
    person : 14
    pottedplant : 15
    sheep : 16
    sofa : 17
    train : 18
    tvmonitor : 19
    background : 20

# discrete labelling functions

# for class 0 ('aeroplane')--------------------------
def lf1d0(image,model1):
    prediction = torch.argmax(model1(image)) #may need to use argmax
    if classes[prediction] == 'aeroplane':
        return 'aeroplane'
    else :
        return ABSTAIN
    
def lf2d0(image,model2):
    prediction = torch.argmax(model2(image))
    if classes[prediction] == 'aeroplane':
        return 'aeroplane'
    else :
        return ABSTAIN
    
def lf3d0(image,model3):
    prediction = torch.argmax(model3(image))
    if classes[prediction] == 'aeroplane':
        return 'aeroplane'
    else :
        return ABSTAIN
    
def lf4d0(image,model4):
    prediction = torch.argmax(model4(image))
    if classes[prediction] == 'aeroplane':
        return 'aeroplane'
    else :
        return ABSTAIN
    
# for class 1 ('bicycle')--------------------------
def lf1d1(image,model1):
    prediction = torch.argmax(model1(image))
    if classes[prediction] == 'bicycle':
        return 'bicycle'
    else :
        return ABSTAIN
    
def lf2d1(image,model2):
    prediction = torch.argmax(model2(image))
    if classes[prediction] == 'bicycle':
        return 'bicycle'
    else :
        return ABSTAIN
    
def lf3d1(image,model3):
    prediction = torch.argmax(model3(image))
    if classes[prediction] == 'bicycle':
        return 'bicycle'
    else :
        return ABSTAIN
    
def lf4d1(image,model4):
    prediction = torch.argmax(model4(image))
    if classes[prediction] == 'bicycle':
        return 'bicycle'
    else :
        return ABSTAIN
    
# for class 2 ('bird')--------------------------
def lf1d2(image,model1):
    prediction = torch.argmax(model1(image))
    if classes[prediction] == 'bird':
        return 'bird'
    else :
        return ABSTAIN
    
def lf2d2(image,model2):
    prediction = torch.argmax(model2(image))
    if classes[prediction] == 'bird':
        return 'bird'
    else :
        return ABSTAIN
    
def lf3d2(image,model3):
    prediction = torch.argmax(model3(image))
    if classes[prediction] == 'bird':
        return 'bird'
    else :
        return ABSTAIN
    
def lf4d2(image,model4):
    prediction = torch.argmax(model4(image))
    if classes[prediction] == 'bird':
        return 'bird'
    else :
        return ABSTAIN
    
# for class 3 ('boat')--------------------------
def lf1d3(image,model1):
    prediction = torch.argmax(model1(image))
    if classes[prediction] == 'boat':
        return 'boat'
    else :
        return ABSTAIN
    
def lf2d3(image,model2):
    prediction = torch.argmax(model2(image))
    if classes[prediction] == 'boat':
        return 'boat'
    else :
        return ABSTAIN
    
def lf3d3(image,model3):
    prediction = torch.argmax(model3(image))
    if classes[prediction] == 'boat':
        return 'boat'
    else :
        return ABSTAIN
    
def lf4d3(image,model4):
    prediction = torch.argmax(model4(image))
    if classes[prediction] == 'boat':
        return 'boat'
    else :
        return ABSTAIN

# for class 4 ('bottle')--------------------------
def lf1d4(image,model1):
    prediction = torch.argmax(model1(image))
    if classes[prediction] == 'bottle':
        return 'bottle'
    else :
        return ABSTAIN
    
def lf2d4(image,model2):
    prediction = torch.argmax(model2(image))
    if classes[prediction] == 'bottle':
        return 'bottle'
    else :
        return ABSTAIN
    
def lf3d4(image,model3):
    prediction = torch.argmax(model3(image))
    if classes[prediction] == 'bottle':
        return 'bottle'
    else :
        return ABSTAIN
    
def lf4d4(image,model4):
    prediction = torch.argmax(model4(image))
    if classes[prediction] == 'bottle':
        return 'bottle'
    else :
        return ABSTAIN
    
# for class 5 ('bus')--------------------------
def lf1d5(image,model1):
    prediction = torch.argmax(model1(image))
    if classes[prediction] == 'bus':
        return 'bus'
    else :
        return ABSTAIN
    
def lf2d5(image,model2):
    prediction = torch.argmax(model2(image))
    if classes[prediction] == 'bus':
        return 'bus'
    else :
        return ABSTAIN
    
def lf3d5(image,model3):
    prediction = torch.argmax(model3(image))
    if classes[prediction] == 'bus':
        return 'bus'
    else :
        return ABSTAIN
    
def lf4d5(image,model4):
    prediction = torch.argmax(model4(image))
    if classes[prediction] == 'bus':
        return 'bus'
    else :
        return ABSTAIN

# for class 6 ('car')
def lf1d6(image, model1):
    prediction = torch.argmax(model1(image))
    if classes[prediction] == 'car':
        return 'car'
    else:
        return ABSTAIN

def lf2d6(image, model2):
    prediction = torch.argmax(model2(image))
    if classes[prediction] == 'car':
        return 'car'
    else:
        return ABSTAIN

def lf3d6(image, model3):
    prediction = torch.argmax(model3(image))
    if classes[prediction] == 'car':
        return 'car'
    else:
        return ABSTAIN

def lf4d6(image, model4):
    prediction = torch.argmax(model4(image))
    if classes[prediction] == 'car':
        return 'car'
    else:
        return ABSTAIN
    

def lf1d7(image, model1):
    prediction = torch.argmax(model1(image))
    if classes[prediction] == 'cat':
        return 'cat'
    else:
        return ABSTAIN

def lf2d7(image, model2):
    prediction = torch.argmax(model2(image))
    if classes[prediction] == 'cat':
        return 'cat'
    else:
        return ABSTAIN

def lf3d7(image, model3):
    prediction = torch.argmax(model3(image))
    if classes[prediction] == 'cat':
        return 'cat'
    else:
        return ABSTAIN

def lf4d7(image, model4):
    prediction = torch.argmax(model4(image))
    if classes[prediction] == 'cat':
        return 'cat'
    else:
        return ABSTAIN

def lf1d8(image, model1):
    prediction = torch.argmax(model1(image))
    if classes[prediction] == 'chair':
        return 'chair'
    else:
        return ABSTAIN

def lf2d8(image, model2):
    prediction = torch.argmax(model2(image))
    if classes[prediction] == 'chair':
        return 'chair'
    else:
        return ABSTAIN

def lf3d8(image, model3):
    prediction = torch.argmax(model3(image))
    if classes[prediction] == 'chair':
        return 'chair'
    else:
        return ABSTAIN

def lf4d8(image, model4):
    prediction = torch.argmax(model4(image))
    if classes[prediction] == 'chair':
        return 'chair'
    else:
        return ABSTAIN

def lf1d9(image, model1):
    prediction = torch.argmax(model1(image))
    if classes[prediction] == 'cow':
        return 'cow'
    else:
        return ABSTAIN

def lf2d9(image, model2):
    prediction = torch.argmax(model2(image))
    if classes[prediction] == 'cow':
        return 'cow'
    else:
        return ABSTAIN

def lf3d9(image, model3):
    prediction = torch.argmax(model3(image))
    if classes[prediction] == 'cow':
        return 'cow'
    else:
        return ABSTAIN

def lf4d9(image, model4):
    prediction = torch.argmax(model4(image))
    if classes[prediction] == 'cow':
        return 'cow'
    else:
        return ABSTAIN

def lf1d10(image, model1):
    prediction = torch.argmax(model1(image))
    if classes[prediction] == 'diningtable':
        return 'diningtable'
    else:
        return ABSTAIN

def lf2d10(image, model2):
    prediction = torch.argmax(model2(image))
    if classes[prediction] == 'diningtable':
        return 'diningtable'
    else:
        return ABSTAIN

def lf3d10(image, model3):
    prediction = torch.argmax(model3(image))
    if classes[prediction] == 'diningtable':
        return 'diningtable'
    else:
        return ABSTAIN

def lf4d10(image, model4):
    prediction = torch.argmax(model4(image))
    if classes[prediction] == 'diningtable':
        return 'diningtable'
    else:
        return ABSTAIN

def lf1d11(image, model1):
    prediction = torch.argmax(model1(image))
    if classes[prediction] == 'dog':
        return 'dog'
    else:
        return ABSTAIN

def lf2d11(image, model2):
    prediction = torch.argmax(model2(image))
    if classes[prediction] == 'dog':
        return 'dog'
    else:
        return ABSTAIN

def lf3d11(image, model3):
    prediction = torch.argmax(model3(image))
    if classes[prediction] == 'dog':
        return 'dog'
    else:
        return ABSTAIN

def lf4d11(image, model4):
    prediction = torch.argmax(model4(image))
    if classes[prediction] == 'dog':
        return 'dog'
    else:
        return ABSTAIN

def lf1d12(image, model1):
    prediction = torch.argmax(model1(image))
    if classes[prediction] == 'horse':
        return 'horse'
    else:
        return ABSTAIN

def lf2d12(image, model2):
    prediction = torch.argmax(model2(image))
    if classes[prediction] == 'horse':
        return 'horse'
    else:
        return ABSTAIN

def lf3d12(image, model3):
    prediction = torch.argmax(model3(image))
    if classes[prediction] == 'horse':
        return 'horse'
    else:
        return ABSTAIN

def lf4d12(image, model4):
    prediction = torch.argmax(model4(image))
    if classes[prediction] == 'horse':
        return 'horse'
    else:
        return ABSTAIN
    
def lf1d13(image, model1):
    prediction = torch.argmax(model1(image))
    if classes[prediction] == 'motorbike':
        return 'motorbike'
    else:
        return ABSTAIN

def lf2d13(image, model2):
    prediction = torch.argmax(model2(image))
    if classes[prediction] == 'motorbike':
        return 'motorbike'
    else:
        return ABSTAIN

def lf3d13(image, model3):
    prediction = torch.argmax(model3(image))
    if classes[prediction] == 'motorbike':
        return 'motorbike'
    else:
        return ABSTAIN

def lf4d13(image, model4):
    prediction = torch.argmax(model4(image))
    if classes[prediction] == 'motorbike':
        return 'motorbike'
    else:
        return ABSTAIN

def lf1d14(image, model1):
    prediction = torch.argmax(model1(image))
    if classes[prediction] == 'person':
        return 'person'
    else:
        return ABSTAIN

def lf2d14(image, model2):
    prediction = torch.argmax(model2(image))
    if classes[prediction] == 'person':
        return 'person'
    else:
        return ABSTAIN

def lf3d14(image, model3):
    prediction = torch.argmax(model3(image))
    if classes[prediction] == 'person':
        return 'person'
    else:
        return ABSTAIN

def lf4d14(image, model4):
    prediction = torch.argmax(model4(image))
    if classes[prediction] == 'person':
        return 'person'
    else:
        return ABSTAIN

def lf1d15(image, model1):
    prediction = torch.argmax(model1(image))
    if classes[prediction] == 'pottedplant':
        return 'pottedplant'
    else:
        return ABSTAIN

def lf2d15(image, model2):
    prediction = torch.argmax(model2(image))
    if classes[prediction] == 'pottedplant':
        return 'pottedplant'
    else:
        return ABSTAIN

def lf3d15(image, model3):
    prediction = torch.argmax(model3(image))
    if classes[prediction] == 'pottedplant':
        return 'pottedplant'
    else:
        return ABSTAIN

def lf4d15(image, model4):
    prediction = torch.argmax(model4(image))
    if classes[prediction] == 'pottedplant':
        return 'pottedplant'
    else:
        return ABSTAIN

def lf1d16(image, model1):
    prediction = torch.argmax(model1(image))
    if classes[prediction] == 'sheep':
        return 'sheep'
    else:
        return ABSTAIN

def lf2d16(image, model2):
    prediction = torch.argmax(model2(image))
    if classes[prediction] == 'sheep':
        return 'sheep'
    else:
        return ABSTAIN

def lf3d16(image, model3):
    prediction = torch.argmax(model3(image))
    if classes[prediction] == 'sheep':
        return 'sheep'
    else:
        return ABSTAIN

def lf4d16(image, model4):
    prediction = torch.argmax(model4(image))
    if classes[prediction] == 'sheep':
        return 'sheep'
    else:
        return ABSTAIN
    
def lf1d17(image, model1):
    prediction = torch.argmax(model1(image))
    if classes[prediction] == 'sofa':
        return 'sofa'
    else:
        return ABSTAIN

def lf2d17(image, model2):
    prediction = torch.argmax(model2(image))
    if classes[prediction] == 'sofa':
        return 'sofa'
    else:
        return ABSTAIN

def lf3d17(image, model3):
    prediction = torch.argmax(model3(image))
    if classes[prediction] == 'sofa':
        return 'sofa'
    else:
        return ABSTAIN

def lf4d17(image, model4):
    prediction = torch.argmax(model4(image))
    if classes[prediction] == 'sofa':
        return 'sofa'
    else:
        return ABSTAIN

def lf1d18(image, model1):
    prediction = torch.argmax(model1(image))
    if classes[prediction] == 'train':
        return 'train'
    else:
        return ABSTAIN

def lf2d18(image, model2):
    prediction = torch.argmax(model2(image))
    if classes[prediction] == 'train':
        return 'train'
    else:
        return ABSTAIN

def lf3d18(image, model3):
    prediction = torch.argmax(model3(image))
    if classes[prediction] == 'train':
        return 'train'
    else:
        return ABSTAIN

def lf4d18(image, model4):
    prediction = torch.argmax(model4(image))
    if classes[prediction] == 'train':
        return 'train'
    else:
        return ABSTAIN

def lf1d19(image, model1):
    prediction = torch.argmax(model1(image))
    if classes[prediction] == 'tvmonitor':
        return 'tvmonitor'
    else:
        return ABSTAIN

def lf2d19(image, model2):
    prediction = torch.argmax(model2(image))
    if classes[prediction] == 'tvmonitor':
        return 'tvmonitor'
    else:
        return ABSTAIN

def lf3d19(image, model3):
    prediction = torch.argmax(model3(image))
    if classes[prediction] == 'tvmonitor':
        return 'tvmonitor'
    else:
        return ABSTAIN

def lf4d19(image, model4):
    prediction = torch.argmax(model4(image))
    if classes[prediction] == 'tvmonitor':
        return 'tvmonitor'
    else:
        return ABSTAIN

def lf1d20(image,model1):
    prediction = torch.argmax(model1(image)) 
    if classes[prediction] == 'background':
        return 'background'
    else :
        return ABSTAIN
    
def lf2d20(image,model2):
    prediction = torch.argmax(model2(image))
    if classes[prediction] == 'background':
        return 'background'
    else :
        return ABSTAIN
    
def lf3d20(image,model3):
    prediction = torch.argmax(model3(image))
    if classes[prediction] == 'background':
        return 'background'
    else :
        return ABSTAIN
    
def lf4d20(image,model4):
    prediction = torch.argmax(model4(image))
    if classes[prediction] == 'background':
        return 'background'
    else :
        return ABSTAIN

#continous labelling fns

threshold = 0
# for class 0 ('aeroplane')--------------------------
def lf1c0(image,model1):
    prediction = model1(image)
    if prediction[ClassLabels.aeroplane] >= threshold:
        return 'aeroplane'
    else :
        return ABSTAIN
    
def lf2c0(image,model2):
    prediction = model2(image)
    if prediction[ClassLabels.aeroplane] >= threshold:
        return 'aeroplane'
    else :
        return ABSTAIN
    
def lf3c0(image,model3):
    prediction = model3(image)
    if prediction[ClassLabels.aeroplane] >= threshold:
        return 'aeroplane'
    else :
        return ABSTAIN
    
def lf4c0(image,model4):
    prediction = model4(image)
    if prediction[ClassLabels.aeroplane] >= threshold:
        return 'aeroplane'
    else :
        return ABSTAIN
    
# for class 1 ('bicycle')--------------------------
def lf1c1(image,model1):
    prediction = model1(image)
    if prediction[ClassLabels.bicycle] >= threshold:
        return 'bicycle'
    else :
        return ABSTAIN
    
def lf2c1(image,model2):
    prediction = model2(image)
    if prediction[ClassLabels.bicycle] >= threshold:
        return 'bicycle'
    else :
        return ABSTAIN
    
def lf3c1(image,model3):
    prediction = model3(image)
    if prediction[ClassLabels.bicycle] >= threshold:
        return 'bicycle'
    else :
        return ABSTAIN
    
def lf4c1(image,model4):
    prediction = model4(image)
    if prediction[ClassLabels.bicycle] >= threshold:
        return 'bicycle'
    else :
        return ABSTAIN
    
# for class 2 ('bird')--------------------------
def lf1c2(image,model1):
    prediction = model1(image)
    if prediction[ClassLabels.bird] >= threshold:
        return 'bird'
    else :
        return ABSTAIN
    
def lf2c2(image,model2):
    prediction = model2(image)
    if prediction[ClassLabels.bird] >= threshold:
        return 'bird'
    else :
        return ABSTAIN
    
def lf3c2(image,model3):
    prediction = model3(image)
    if prediction[ClassLabels.bird] >= threshold:
        return 'bird'
    else :
        return ABSTAIN
    
def lf4c2(image,model4):
    prediction = model4(image)
    if prediction[ClassLabels.bird] >= threshold:
        return 'bird'
    else :
        return ABSTAIN
    
# for class 3 ('boat')--------------------------
def lf1c3(image,model1):
    prediction = model1(image)
    if prediction[ClassLabels.boat] >= threshold:
        return 'boat'
    else :
        return ABSTAIN
    
def lf2c3(image,model2):
    prediction = model2(image)
    if prediction[ClassLabels.boat] >= threshold:
        return 'boat'
    else :
        return ABSTAIN
    
def lf3c3(image,model3):
    prediction = model3(image)
    if prediction[ClassLabels.boat] >= threshold:
        return 'boat'
    else :
        return ABSTAIN
    
def lf4c3(image,model4):
    prediction = model4(image)
    if prediction[ClassLabels.boat] >= threshold:
        return 'boat'
    else :
        return ABSTAIN

# for class 4 ('bottle')--------------------------
def lf1c4(image,model1):
    prediction = model1(image)
    if prediction[ClassLabels.bottle] >= threshold:
        return 'bottle'
    else :
        return ABSTAIN
    
def lf2c4(image,model2):
    prediction = model2(image)
    if prediction[ClassLabels.bottle] >= threshold:
        return 'bottle'
    else :
        return ABSTAIN
    
def lf3c4(image,model3):
    prediction = model3(image)
    if prediction[ClassLabels.bottle] >= threshold:
        return 'bottle'
    else :
        return ABSTAIN
    
def lf4c4(image,model4):
    prediction = model4(image)
    if prediction[ClassLabels.bottle] >= threshold:
        return 'bottle'
    else :
        return ABSTAIN
    
# for class 5 ('bus')--------------------------
def lf1c5(image,model1):
    prediction = model1(image)
    if prediction[ClassLabels.bus] >= threshold:
        return 'bus'
    else :
        return ABSTAIN
    
def lf2c5(image,model2):
    prediction = model2(image)
    if prediction[ClassLabels.bus] >= threshold:
        return 'bus'
    else :
        return ABSTAIN
    
def lf3c5(image,model3):
    prediction = model3(image)
    if prediction[ClassLabels.bus] >= threshold:
        return 'bus'
    else :
        return ABSTAIN
    
def lf4c5(image,model4):
    prediction = model4(image)
    if prediction[ClassLabels.bus] >= threshold:
        return 'bus'
    else :
        return ABSTAIN

# for class 6 ('car')
def lf1c6(image, model1):
    prediction = model1(image)
    if prediction[ClassLabels.car] >= threshold:
        return 'car'
    else:
        return ABSTAIN

def lf2c6(image, model2):
    prediction = model2(image)
    if prediction[ClassLabels.car] >= threshold:
        return 'car'
    else:
        return ABSTAIN

def lf3c6(image, model3):
    prediction = model3(image)
    if prediction[ClassLabels.car] >= threshold:
        return 'car'
    else:
        return ABSTAIN

def lf4c6(image, model4):
    prediction = model4(image)
    if prediction[ClassLabels.car] >= threshold:
        return 'car'
    else:
        return ABSTAIN
    

def lf1c7(image, model1):
    prediction = model1(image)
    if prediction[ClassLabels.cat] >= threshold:
        return 'cat'
    else:
        return ABSTAIN

def lf2c7(image, model2):
    prediction = model2(image)
    if prediction[ClassLabels.cat] >= threshold:
        return 'cat'
    else:
        return ABSTAIN

def lf3c7(image, model3):
    prediction = model3(image)
    if prediction[ClassLabels.cat] >= threshold:
        return 'cat'
    else:
        return ABSTAIN

def lf4c7(image, model4):
    prediction = model4(image)
    if prediction[ClassLabels.cat] >= threshold:
        return 'cat'
    else:
        return ABSTAIN

def lf1c8(image, model1):
    prediction = model1(image)
    if prediction[ClassLabels.chair] >= threshold:
        return 'chair'
    else:
        return ABSTAIN

def lf2c8(image, model2):
    prediction = model2(image)
    if prediction[ClassLabels.chair] >= threshold:
        return 'chair'
    else:
        return ABSTAIN

def lf3c8(image, model3):
    prediction = model3(image)
    if prediction[ClassLabels.chair] >= threshold:
        return 'chair'
    else:
        return ABSTAIN

def lf4c8(image, model4):
    prediction = model4(image)
    if prediction[ClassLabels.chair] >= threshold:
        return 'chair'
    else:
        return ABSTAIN

def lf1c9(image, model1):
    prediction = model1(image)
    if prediction[ClassLabels.cow] >= threshold:
        return 'cow'
    else:
        return ABSTAIN

def lf2c9(image, model2):
    prediction = model2(image)
    if prediction[ClassLabels.cow] >= threshold:
        return 'cow'
    else:
        return ABSTAIN

def lf3c9(image, model3):
    prediction = model3(image)
    if prediction[ClassLabels.cow] >= threshold:
        return 'cow'
    else:
        return ABSTAIN

def lf4c9(image, model4):
    prediction = model4(image)
    if prediction[ClassLabels.cow] >= threshold:
        return 'cow'
    else:
        return ABSTAIN

def lf1c10(image, model1):
    prediction = model1(image)
    if prediction[ClassLabels.diningtable] >= threshold:
        return 'diningtable'
    else:
        return ABSTAIN

def lf2c10(image, model2):
    prediction = model2(image)
    if prediction[ClassLabels.diningtable] >= threshold:
        return 'diningtable'
    else:
        return ABSTAIN

def lf3c10(image, model3):
    prediction = model3(image)
    if prediction[ClassLabels.diningtable] >= threshold:
        return 'diningtable'
    else:
        return ABSTAIN

def lf4c10(image, model4):
    prediction = model4(image)
    if prediction[ClassLabels.diningtable] >= threshold:
        return 'diningtable'
    else:
        return ABSTAIN

def lf1c11(image, model1):
    prediction = model1(image)
    if prediction[ClassLabels.dog] >= threshold:
        return 'dog'
    else:
        return ABSTAIN

def lf2d11(image, model2):
    prediction = model2(image)
    if prediction[ClassLabels.dog] >= threshold:
        return 'dog'
    else:
        return ABSTAIN

def lf3c11(image, model3):
    prediction = model3(image)
    if prediction[ClassLabels.dog] >= threshold:
        return 'dog'
    else:
        return ABSTAIN

def lf4c11(image, model4):
    prediction = model4(image)
    if prediction[ClassLabels.dog] >= threshold:
        return 'dog'
    else:
        return ABSTAIN

def lf1c12(image, model1):
    prediction = model1(image)
    if prediction[ClassLabels.horse] >= threshold:
        return 'horse'
    else:
        return ABSTAIN

def lf2c12(image, model2):
    prediction = model2(image)
    if prediction[ClassLabels.horse] >= threshold:
        return 'horse'
    else:
        return ABSTAIN

def lf3c12(image, model3):
    prediction = model3(image)
    if prediction[ClassLabels.horse] >= threshold:
        return 'horse'
    else:
        return ABSTAIN

def lf4c12(image, model4):
    prediction = model4(image)
    if prediction[ClassLabels.horse] >= threshold:
        return 'horse'
    else:
        return ABSTAIN
    
def lf1c13(image, model1):
    prediction = model1(image)
    if prediction[ClassLabels.motorbike] >= threshold:
        return 'motorbike'
    else:
        return ABSTAIN

def lf2c13(image, model2):
    prediction = model2(image)
    if prediction[ClassLabels.motorbike] >= threshold:
        return 'motorbike'
    else:
        return ABSTAIN

def lf3c13(image, model3):
    prediction = model3(image)
    if prediction[ClassLabels.motorbike] >= threshold:
        return 'motorbike'
    else:
        return ABSTAIN

def lf4c13(image, model4):
    prediction = model4(image)
    if prediction[ClassLabels.motorbike] >= threshold:
        return 'motorbike'
    else:
        return ABSTAIN

def lf1c14(image, model1):
    prediction = model1(image)
    if prediction[ClassLabels.person] >= threshold:
        return 'person'
    else:
        return ABSTAIN

def lf2c14(image, model2):
    prediction = model2(image)
    if prediction[ClassLabels.person] >= threshold:
        return 'person'
    else:
        return ABSTAIN

def lf3c14(image, model3):
    prediction = model3(image)
    if prediction[ClassLabels.person] >= threshold:
        return 'person'
    else:
        return ABSTAIN

def lf4c14(image, model4):
    prediction = model4(image)
    if prediction[ClassLabels.person] >= threshold:
        return 'person'
    else:
        return ABSTAIN

def lf1c15(image, model1):
    prediction = model1(image)
    if prediction[ClassLabels.pottedplant] >= threshold:
        return 'pottedplant'
    else:
        return ABSTAIN

def lf2c15(image, model2):
    prediction = model2(image)
    if prediction[ClassLabels.pottedplant] >= threshold:
        return 'pottedplant'
    else:
        return ABSTAIN

def lf3c15(image, model3):
    prediction = model3(image)
    if prediction[ClassLabels.pottedplant] >= threshold:
        return 'pottedplant'
    else:
        return ABSTAIN

def lf4c15(image, model4):
    prediction = model4(image)
    if prediction[ClassLabels.pottedplant] >= threshold:
        return 'pottedplant'
    else:
        return ABSTAIN

def lf1c16(image, model1):
    prediction = model1(image)
    if prediction[ClassLabels.sheep] >= threshold:
        return 'sheep'
    else:
        return ABSTAIN

def lf2c16(image, model2):
    prediction = model2(image)
    if prediction[ClassLabels.sheep] >= threshold:
        return 'sheep'
    else:
        return ABSTAIN

def lf3c16(image, model3):
    prediction = model3(image)
    if prediction[ClassLabels.sheep] >= threshold:
        return 'sheep'
    else:
        return ABSTAIN

def lf4c16(image, model4):
    prediction = model4(image)
    if prediction[ClassLabels.sheep] >= threshold:
        return 'sheep'
    else:
        return ABSTAIN
    
def lf1c17(image, model1):
    prediction = model1(image)
    if prediction[ClassLabels.sofa] >= threshold:
        return 'sofa'
    else:
        return ABSTAIN

def lf2c17(image, model2):
    prediction = model2(image)
    if prediction[ClassLabels.sofa] >= threshold:
        return 'sofa'
    else:
        return ABSTAIN

def lf3c17(image, model3):
    prediction = model3(image)
    if prediction[ClassLabels.sofa] >= threshold:
        return 'sofa'
    else:
        return ABSTAIN

def lf4c17(image, model4):
    prediction = model4(image)
    if prediction[ClassLabels.sofa] >= threshold:
        return 'sofa'
    else:
        return ABSTAIN

def lf1c18(image, model1):
    prediction = model1(image)
    if prediction[ClassLabels.train] >= threshold:
        return 'train'
    else:
        return ABSTAIN

def lf2c18(image, model2):
    prediction = model2(image)
    if prediction[ClassLabels.train] >= threshold:
        return 'train'
    else:
        return ABSTAIN

def lf3c18(image, model3):
    prediction = model3(image)
    if prediction[ClassLabels.train] >= threshold:
        return 'train'
    else:
        return ABSTAIN

def lf4c18(image, model4):
    prediction = model4(image)
    if prediction[ClassLabels.train] >= threshold:
        return 'train'
    else:
        return ABSTAIN

def lf1c19(image, model1):
    prediction = model1(image)
    if prediction[ClassLabels.tvmonitor] >= threshold:
        return 'tvmonitor'
    else:
        return ABSTAIN

def lf2c19(image, model2):
    prediction = model2(image)
    if prediction[ClassLabels.tvmonitor] >= threshold:
        return 'tvmonitor'
    else:
        return ABSTAIN

def lf3c19(image, model3):
    prediction = model3(image)
    if prediction[ClassLabels.tvmonitor] >= threshold:
        return 'tvmonitor'
    else:
        return ABSTAIN

def lf4c19(image, model4):
    prediction = model4(image)
    if prediction[ClassLabels.tvmonitor] >= threshold:
        return 'tvmonitor'
    else:
        return ABSTAIN   
    
def lf1c20(image, model1):
    prediction = model1(image)
    if prediction[ClassLabels.background] >= threshold:
        return 'background'
    else:
        return ABSTAIN

def lf2c20(image, model2):
    prediction = model2(image)
    if prediction[ClassLabels.background] >= threshold:
        return 'background'
    else:
        return ABSTAIN

def lf3c20(image, model3):
    prediction = model3(image)
    if prediction[ClassLabels.background] >= threshold:
        return 'background'
    else:
        return ABSTAIN

def lf4c20(image, model4):
    prediction = model4(image)
    if prediction[ClassLabels.background] >= threshold:
        return 'background'
    else:
        return ABSTAIN   
    
a = ['c','d']
LFS = []

for c in a:
    for i in range(4):
        for j in range(21):
            chr = 'lf' + str(i+1) + c + str(j)
            LFS.append(chr)

rules = LFSet("RULES_LF")
rules.add_lf_list(LFS)

