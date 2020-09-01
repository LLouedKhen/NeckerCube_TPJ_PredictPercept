#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 16:00:09 2019

This script employs nilearn. I wrote this to determine if a specific brain region 
BOLD response can predict a known brain state (in this case, switching into a 
from-above Necker Cube percept.)
If you want to change the predicting region, change the mask in line 236

Also, note that beta images come from a prior LSS estimation procedure.

*Probably a lot of redundant stuff too but, you know what, I'm a mom.*

@author: loued
"""

import pandas as pd
import os 
import numpy as np
import glob
import pickle 
import matplotlib 
from nilearn.input_data import NiftiMasker
from nilearn import plotting
from sklearn.model_selection import permutation_test_score   
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score    
from sklearn.metrics import accuracy_score
import nilearn.decoding
from nilearn.plotting import plot_stat_map, plot_img, show
import matplotlib 
matplotlib.use("module://ipykernel.pylab.backend_inline")

dataPath = '/Users/loued/Documents/PerceptualUncertaintyLSS'
os.chdir(dataPath)

UpBetas = []
DownBetas = []
AllBetas = []
behavioral = []
betasAll = []

SubjectList = glob.glob('DMUC*')
for i in range(0,len(SubjectList)):
        os.chdir(SubjectList[i])
        PR = glob.glob('PR0*')
        SubjectFULLpath = os.path.join(dataPath, SubjectList[i], PR[0])
        #common path after PR number is 
        commonPath = 'StatsNCSTC_KPSurp_OldSwitchSurpOnlyUPDOWN/LSS/betas/Sess/001'
        thisSubjectPathtoLSS = os.path.join(SubjectFULLpath, commonPath)
        os.chdir(thisSubjectPathtoLSS)
        thisUpBetas = glob.glob('Up*.nii')
        for j in range(0,len(thisUpBetas)):
            thisUpBetas[j] = os.path.join(thisSubjectPathtoLSS, thisUpBetas[j])
        thisListu = ['Up'] * len(thisUpBetas)
        UpBetas.append(thisUpBetas)
        AllBetas.append(thisUpBetas)
        #behavioral.append(thisList)
        thisDownBetas = glob.glob('Down*.nii')
        for k in range(0,len(thisDownBetas)):
            thisDownBetas[k] = os.path.join(thisSubjectPathtoLSS, thisDownBetas[k])
        thisListd = ['Down'] * len(thisDownBetas)
        DownBetas.append(thisDownBetas)
        AllBetas.append(thisDownBetas)      
        behavioral.append(thisListu + thisListd)
        betasAll.append(thisUpBetas + thisDownBetas)
        os.chdir(dataPath)
        
TrueUpBetas = UpBetas

for i in range(0,len(SubjectList)):
    UpBetas[i].extend(DownBetas[i])
    
AllBetasS = UpBetas
UpBetas = TrueUpBetas
        
os.chdir(dataPath)
     
Masks =[[] for i in range(21)]   
Mask = []
for i in range(0,len(SubjectList)):
    os.chdir(SubjectList[i])
    PR = glob.glob('PR0*')
    SubjectFULLpath = os.path.join(dataPath, SubjectList[i], PR[0])
    #common path after PR number is 
    commonPath = 'StatsNCSTC_KPSurp_OldSwitchSurpOnlyUPDOWN/LSS/betas/Sess/001'
    thisSubjectPath = os.path.join(SubjectFULLpath, commonPath)
    os.chdir(thisSubjectPath)
    #Masks[i] = glob.glob('mask.nii', recursive=True)
    subPath = [x[0] for x in os.walk(thisSubjectPath)]
    for j in range(1, len(subPath)):
        thisMask = os.path.join(thisSubjectPath,subPath[j], 'mask.nii')
        Masks[i].append(thisMask)
    thisMeanMask = nilearn.masking.compute_epi_mask(Masks[i])
    Mask.append(thisMeanMask)
    os.chdir(dataPath)
        
bigListUp = []
bigListMaskUp = []
for i in range(0,len(UpBetas)):
    for j in range(0,len(UpBetas[i])):
        bigListUp.append(UpBetas[i][j])
        bigListMaskUp.append(Masks[i])
        
bigListDown= []
bigListMaskDown = []
for i in range(0,len(DownBetas)):
    for j in range(0,len(DownBetas[i])):
        bigListDown.append(DownBetas[i][j])
        bigListMaskDown.append(Masks[i])
        
  
fMRI_masked=[[] for i in range(21)]    
Maskers = []

for i in range(0,len(AllBetasS)):
    thisMasker = NiftiMasker(mask_img=Mask[i])
    Maskers.append(thisMasker)
    for j in range (0, len(AllBetasS[i])):
        thisfMRI_masked= thisMasker.fit_transform([AllBetasS[i][j]])
        fMRI_masked[i].append(thisfMRI_masked)
    

#####Whole Brain ######

from sklearn.svm import SVC
svc = SVC(C=1., kernel='linear')

myFits = [[] for u in range(21)]

for i in range(0,21):
    thisSet = np.squeeze(np.array(fMRI_masked[i]))
    myFits[i] = svc.fit(thisSet, behavioral[i])
    thisFile = 'myFitsSVC' + str(i) + '.sav'
    pickle.dump(myFits[i], open(thisFile, 'wb'))


from sklearn.model_selection import KFold

cv = KFold(n_splits=5)

conditions_masked = [[] for u in range(21)]
fits = [[] for u in range(21)]
prediction = [[] for u in range(21)]
conditions = [[] for u in range(21)]

for i in range(0,21):
    for j in range(0, len(behavioral[i])):
        if behavioral[i][j] == 'Up':
            conditions[i].append(1)
        elif behavioral[i][j] == 'Down':
            conditions[i].append(-1)



# Make processing parallel
# /!\ As each thread will print its progress, n_jobs > 1 could mess up the
#     information output.
n_jobs = 1

# Define the cross-validation scheme used for validation.
# Here we use a KFold cross-validation on the session, which corresponds to
# splitting the samples in 4 folds and make 4 runs using each fold as a test
# set once and the others as learning sets
from sklearn.model_selection import KFold
cv = KFold(n_splits=4)

import time

classifiers_scores_Pred= [[] for u in range(21)]
classifiers_scores_f1 = [[] for u in range(21)]
score = [[] for u in range(21)]
permutation_scores = [[] for u in range(21)]
pvalue =[[] for u in range(21)]


for i in range(0,21):
    for train, test in cv.split(X=fMRI_masked[i]):
        conditions_masked[i] = np.array(conditions[i])[train]
        fits[i]= svc.fit(np.squeeze(np.array(fMRI_masked[i]))[train], conditions_masked[i])
        prediction[i] = svc.predict(np.squeeze(np.array(fMRI_masked[i]))[test])
        classifiers_scores_Pred[i] = accuracy_score(np.array(conditions[i])[test], prediction[i])
        classifiers_scores_f1[i] = cross_val_score(svc, np.squeeze(np.array(fMRI_masked[i])), conditions[i], scoring = 'f1')
        #np.array(conditions[i]),cv=cv, scoring="f1")
        score[i], permutation_scores[i], pvalue[i] = permutation_test_score(svc, np.squeeze(np.array(fMRI_masked[i])), np.array(conditions[i]), scoring="accuracy", cv=cv, n_permutations=10, n_jobs=1)
        print((prediction[i] == np.array(conditions[i])[test]).sum()
             / float(len(np.array(conditions[i])[test])))
        
from nilearn.image import new_img_like, load_img, get_data
from nilearn.input_data import NiftiMasker


#############
from sklearn.feature_selection import f_classif
f_values =[[] for u in range(0, 21)] 
p_values =[[] for u in range(0, 21)] 
p_unmasked =[[] for u in range(0, 21)] 
f_unmasked =[[] for u in range(0, 21)] 
sigP =[[] for u in range(0, 21)]
sigF = [[] for u in range(0, 21)]

canImg =[]

for i in range(0,21):
    f_values[i], p_values[i] = f_classif(fMRI_masked[i], conditions[i])
    sigP[i] = p_values[i] <0.05
    p_values[i] = -np.log10(p_values[i])
    p_values[i][p_values[i] > 10] = 10
    sigF[i]=  f_values[i] * sigP[i]
    p_unmasked[i] = get_data(Maskers[i].inverse_transform(p_values[i]))
    f_unmasked[i] = get_data(Maskers[i].inverse_transform(sigF[i]))
#for i in range(0,21):
#   canImg[i] = nilearn.image.mean_img(AllBetasS[i])
    
canImg = '/Users/loued/Documents/MATLAB/spm12/canonical/single_subj_T1.nii'

p_ma = [[] for u in range(0, 21)] 
f_score_img = [[] for u in range(0, 21)] 
f_ma = [[] for u in range(0, 21)] 


for i in range(0,21):

    p_ma[i] = np.ma.array(p_unmasked[i])
    f_ma[i] = np.ma.array(f_unmasked[i])
    f_score_img[i] = new_img_like(canImg, f_ma[i])
    display= plot_stat_map(f_score_img[i], canImg,
              title="F-scores", display_mode="ortho",
              colorbar=False)

    show()
    display.close()
    thisImg = f_score_img[i]
    thisImg.to_filename('f_score_img_NCLSS_' + str(i) + '.nii.gz')
    
##############RB MARS rTPJ

rTPJMask = '/Users/loued/Documents/PerceptualUncertaintyLSS/MarsTPJParcellation/TPJ_thr25_1mmDorsal.nii'   

from sklearn.svm import SVC
svc = SVC(C=1., kernel='linear')

myFits = [[] for u in range(21)]

for i in range(0,21):
    thisSet = np.squeeze(np.array(fMRI_masked[i]))
    myFits[i] = svc.fit(thisSet, behavioral[i])
    thisFile = 'myFitsSVC' + str(i) + '.sav'
    pickle.dump(myFits[i], open(thisFile, 'wb'))


from sklearn.model_selection import KFold

cv = KFold(n_splits=5)

conditions_masked = [[] for u in range(21)]
fits = [[] for u in range(21)]
prediction = [[] for u in range(21)]
conditions = [[] for u in range(21)]

for i in range(0,21):
    for j in range(0, len(behavioral[i])):
        if behavioral[i][j] == 'Up':
            conditions[i].append(1)
        elif behavioral[i][j] == 'Down':
            conditions[i].append(-1)



# Make processing parallel
# /!\ As each thread will print its progress, n_jobs > 1 could mess up the
#     information output.
n_jobs = 1

# Define the cross-validation scheme used for validation.
# Here we use a KFold cross-validation on the session, which corresponds to
# splitting the samples in 4 folds and make 4 runs using each fold as a test
# set once and the others as learning sets
from sklearn.model_selection import KFold
cv = KFold(n_splits=4)

import time

classifiers_scores_Pred= [[] for u in range(21)]
classifiers_scores_f1 = [[] for u in range(21)]
score = [[] for u in range(21)]
permutation_scores = [[] for u in range(21)]
pvalue =[[] for u in range(21)]


for i in range(0,21):
    for train, test in cv.split(X=fMRI_masked[i]):
        conditions_masked[i] = np.array(conditions[i])[train]
        fits[i]= svc.fit(np.squeeze(np.array(fMRI_masked[i]))[train], conditions_masked[i])
        prediction[i] = svc.predict(np.squeeze(np.array(fMRI_masked[i]))[test])
        classifiers_scores_Pred[i] = accuracy_score(np.array(conditions[i])[test], prediction[i])
        classifiers_scores_f1[i] = cross_val_score(svc, np.squeeze(np.array(fMRI_masked[i])), conditions[i], scoring = 'f1')
        #np.array(conditions[i]),cv=cv, scoring="f1")
        score[i], permutation_scores[i], pvalue[i] = permutation_test_score(svc, np.squeeze(np.array(fMRI_masked[i])), np.array(conditions[i]), scoring="accuracy", cv=cv, n_permutations=10, n_jobs=1)
        print((prediction[i] == np.array(conditions[i])[test]).sum()
             / float(len(np.array(conditions[i])[test])))
        
from nilearn.image import new_img_like, load_img, get_data
from nilearn.input_data import NiftiMasker


#############
from sklearn.feature_selection import f_classif
f_values =[[] for u in range(0, 21)] 
p_values =[[] for u in range(0, 21)] 
p_unmasked =[[] for u in range(0, 21)] 
f_unmasked =[[] for u in range(0, 21)] 
sigP =[[] for u in range(0, 21)]
sigF = [[] for u in range(0, 21)]

canImg =[]

for i in range(0,21):
    f_values[i], p_values[i] = f_classif(fMRI_masked[i], conditions[i])
    sigP[i] = p_values[i] <0.05
    p_values[i] = -np.log10(p_values[i])
    p_values[i][p_values[i] > 10] = 10
    sigF[i]=  f_values[i] * sigP[i]
    p_unmasked[i] = get_data(Maskers[i].inverse_transform(p_values[i]))
    f_unmasked[i] = get_data(Maskers[i].inverse_transform(sigF[i]))
#for i in range(0,21):
#   canImg[i] = nilearn.image.mean_img(AllBetasS[i])
    
canImg = '/Users/loued/Documents/MATLAB/spm12/canonical/single_subj_T1.nii'

p_ma = [[] for u in range(0, 21)] 
f_score_img = [[] for u in range(0, 21)] 
f_ma = [[] for u in range(0, 21)] 


for i in range(0,21):

    p_ma[i] = np.ma.array(p_unmasked[i])
    f_ma[i] = np.ma.array(f_unmasked[i])
    f_score_img[i] = new_img_like(canImg, f_ma[i])
    display= plot_stat_map(f_score_img[i], canImg,
              title="F-scores", display_mode="ortho",
              colorbar=False)

    show()
    display.close()
    thisImg = f_score_img[i]
    thisImg.to_filename('f_score_img_NCLSS_rTPJ_' + str(i) + '.nii.gz')
    

    
    
