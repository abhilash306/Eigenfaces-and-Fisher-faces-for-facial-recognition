# Eigenfaces-and-Fisher-faces-for-facial-recognition

---
 Abhilash 

---

## Index
- Introduction
- Dataset Explanation
- Eigenfaces For 2 Classes
- Eigenfaces for Multiple Classes
- Fisherfaces for Multiple Classes

---

## Introduction:
In this assignment, we'll be delving deep into two fundamental concepts of machine learning: eigenspaces in unsupervised learning and Fisher's linear discriminant in supervised learning. Our adventure begins with a 2-class problem, where we'll be dealing with different views of faces belonging to two separate individuals. Our goal is to train a model to distinguish between these individuals accurately.

In the supervised case, our mission is to identify the best separating hyperplane that effectively divides the two classes of faces. On the other hand, in the unsupervised scenario, we'll embark on the challenge of discovering the corresponding decision boundary without the luxury of explicit class labels. Throughout this assignment, we'll explore techniques such as analyzing cluster centers and distances to enhance our understanding and application of these concepts. Additionally, to ensure robust evaluation, we'll meticulously divide our dataset into distinct training and testing sets. So, let's embark on this captivating journey, leveraging any face recognition database of our choice, to unravel the secrets of facial recognition through the lens of machine learning expertise.

---

## Dataset Explanation:
- The AT&T Laboratories Cambridge Face Database comprises facial images captured between April 1992 and April 1994.
- Each of the 40 distinct subjects in the database is represented by 10 different images.
- Images within the database vary in terms of lighting conditions, facial expressions, and facial details, including the presence or absence of glasses.
- Each image in the database measures 92x112 pixels and contains 256 grey levels per pixel.
- With a total of 92 * 112 = 10,304 features per image, this database offers a rich resource for exploring facial recognition algorithms and techniques.

---

## Eigenfaces for 2 Classes:
We have taken 6 training images and 4 testing images.

### Training images:
**Subject 1:**

![Subject 1 Training Image 1]![image](https://github.com/abhilash306/Eigenfaces-and-Fisher-faces-for-facial-recognition/assets/29005113/9e16b878-a7e1-414d-8cac-52f72b9384ff)

![Subject 1 Training Image 2]![image](https://github.com/abhilash306/Eigenfaces-and-Fisher-faces-for-facial-recognition/assets/29005113/f8f2d30b-271f-4803-b54f-8d962b5ecb98)

![Subject 1 Training Image 3]![image](https://github.com/abhilash306/Eigenfaces-and-Fisher-faces-for-facial-recognition/assets/29005113/ad773919-a666-4d0d-bf23-86da9267d3f1)
 
![Subject 1 Training Image 4]![image](https://github.com/abhilash306/Eigenfaces-and-Fisher-faces-for-facial-recognition/assets/29005113/9ca37161-6a71-4dae-a694-0a6a85239bb1)
  
![Subject 1 Training Image 5]![image](https://github.com/abhilash306/Eigenfaces-and-Fisher-faces-for-facial-recognition/assets/29005113/f6608fb8-f4b4-4aac-9f7e-15b5c396a265)
  
![Subject 1 Training Image 6]![image](https://github.com/abhilash306/Eigenfaces-and-Fisher-faces-for-facial-recognition/assets/29005113/5563e319-1327-4a2f-bffc-fccafacba412)


**Subject 2:**

![Subject 2 Training Image 1]![image](https://github.com/abhilash306/Eigenfaces-and-Fisher-faces-for-facial-recognition/assets/29005113/dff7c98e-a828-4990-b30f-c8d95603a747)

![Subject 2 Training Image 2]![image](https://github.com/abhilash306/Eigenfaces-and-Fisher-faces-for-facial-recognition/assets/29005113/382b16b7-8cd2-4228-be26-1ccabf009440)

![Subject 2 Training Image 3]![image](https://github.com/abhilash306/Eigenfaces-and-Fisher-faces-for-facial-recognition/assets/29005113/617b87d3-98f0-47fe-aa50-f7c17c5b6f7d)

![Subject 2 Training Image 4]![image](https://github.com/abhilash306/Eigenfaces-and-Fisher-faces-for-facial-recognition/assets/29005113/440c64f9-38eb-4e53-8352-665f341823a9)

![Subject 2 Training Image 5]![image](https://github.com/abhilash306/Eigenfaces-and-Fisher-faces-for-facial-recognition/assets/29005113/f306d3d8-2ccc-4192-a263-8285d09f0ae5)

![Subject 2 Training Image 6]![image](https://github.com/abhilash306/Eigenfaces-and-Fisher-faces-for-facial-recognition/assets/29005113/805b67cd-18e6-4551-82b4-8f06e732e76f)


### Testing images:

![Subject 1 Testing Image 1]![image](https://github.com/abhilash306/Eigenfaces-and-Fisher-faces-for-facial-recognition/assets/29005113/c0c69688-1e82-4d22-9a25-c6091429a607)

![Subject 1 Testing Image 2]![image](https://github.com/abhilash306/Eigenfaces-and-Fisher-faces-for-facial-recognition/assets/29005113/f7e4d2d2-2896-44c4-a86a-ef792a73d22a)
 
![Subject 2 Testing Image 1]![image](https://github.com/abhilash306/Eigenfaces-and-Fisher-faces-for-facial-recognition/assets/29005113/f1403ded-6ece-42a1-8b22-af8de0d531a6)

![Subject 2 Testing Image 2]![image](https://github.com/abhilash306/Eigenfaces-and-Fisher-faces-for-facial-recognition/assets/29005113/e5f2f4da-8c0d-470b-85e6-15a85ac94455)


### Eigenfaces for Face Recognition:
To implement Eigenfaces for face recognition, the first step is to split the available data into training and test sets. From the training set of face images, the next step involves calculating the eigenfaces, which essentially define the face space using principal component analysis (PCA).

When encountering a new face image, a set of weights is computed based on the input image and the M eigenfaces by projecting the input image onto each of the eigenfaces. Subsequently, the proximity of the image to the "face space" is evaluated to determine if it's a face, whether known or unknown. Finally, the weight pattern is classified either as a known person or as unknown based on predefined criteria. This methodology forms the core of Eigenfaces for effective face recognition.

---

## Principal Component Analysis:
Principal Component Analysis (PCA) is a statistical technique used to simplify high-dimensional datasets by identifying patterns and reducing dimensionality. It computes the covariance matrix, then extracts eigenvectors and eigenvalues via eigenvalue decomposition. These eigenvectors, representing directions of maximum variance, are sorted by eigenvalues, indicating variance explained. By selecting top principal components, PCA reduces dataset dimensions while preserving most variance. Widely applied in image processing, data compression, and pattern recognition, PCA aids in feature extraction, noise reduction, and data visualization. Its versatility enables efficient analysis and interpretation of complex datasets, fostering insights and enhancing decision-making across various disciplines.

### Mean Face & Normalized Training Images:

![Mean Face]![image](https://github.com/abhilash306/Eigenfaces-and-Fisher-faces-for-facial-recognition/assets/29005113/1995fc2f-a11b-4493-8ba9-29b105a1d3dd)


### Eigenfaces (2 out of 10304):

![Eigenface 1]![image](https://github.com/abhilash306/Eigenfaces-and-Fisher-faces-for-facial-recognition/assets/29005113/5e650d24-ade8-4bbe-8e40-afd00b450538)



### PCA result:

![PCA result]![image](https://github.com/abhilash306/Eigenfaces-and-Fisher-faces-for-facial-recognition/assets/29005113/84162fb4-5faa-492e-aa91-4fd2308e74c4)


---

## K-Means Clustering:
Here we can see that the distance of all images of subject 0 from its cluster is less and the distance from other clusters is more. Similarly, for subject 1, the distance of all images of subject 1 from its cluster is less and the distance from other clusters is more.
![image](https://github.com/abhilash306/Eigenfaces-and-Fisher-faces-for-facial-recognition/assets/29005113/5d57b57e-b868-4606-a637-a475a1dcb14d)


### Result:
**Accuracy:** 100%

---

## Eigenfaces for Multi Class:
Here we have taken 10 different classes. For training, a total of 60 images - 6 images for each subject. And for testing, we have taken 40 images, 4 images for each subject.

### Mean and Normalized Training Images:

![Mean and Normalized Training Images](#)![image](https://github.com/abhilash306/Eigenfaces-and-Fisher-faces-for-facial-recognition/assets/29005113/906acb37-1fd8-4273-aee6-61f208cf789c)


### Eigenfaces:

![Eigenface](#)![image](https://github.com/abhilash306/Eigenfaces-and-Fisher-faces-for-facial-recognition/assets/29005113/35be0246-a0a7-4bee-8b4d-5dd44820f09b)
![image](https://github.com/abhilash306/Eigenfaces-and-Fisher-faces-for-facial-recognition/assets/29005113/cbf60aae-a232-49ed-87aa-d13a4cf3384f)



### Result:
**K = 30**

---

## Fisherfaces for Multiple Classes:
1. Split the available data into train and test sets.
2. From the training set of face images, calculate the eigenvalues, which define the eigenspace using principal component analysis.
3. Find the mean vectors of each class from the dataset and an overall mean vector of all the training images.
4. Compute scatter matrices:
   - Between class scatter matrix - SB
   - Within class scatter matrix - SW

### Fisher LDA:
1. J = SB \ SW
2. Compute the eigenvectors and corresponding eigenvalues for the scatter matrices J.
3. Sort the eigenvectors by decreasing eigenvalues and choose k eigenvectors.
4. WFS = WPCA * Reduced data of LDA.
5. When a new face image is encountered, calculate a set of weights based on the input image and the k fisherfaces by projecting the input image onto each of the fisherfaces.
6. Determine if the image is a face (whether known or unknown) by checking to see if the image is sufficiently close to “face space.”
7. Classify the weight pattern as either a known person or as unknown.

### Mean Face:

![Mean Face](#)![image](https://github.com/abhilash306/Eigenfaces-and-Fisher-faces-for-facial-recognition/assets/29005113/57d95018-d8ad-4a40-ad94-e932219fdafc)


### LDA Results for 2 Class Problem:

![LDA Results](#)![image](https://github.com/abhilash306/Eigenfaces-and-Fisher-faces-for-facial-recognition/assets/29005113/e017cbda-2bc4-4b5d-8d72-e0fdceff42ac)


### Result:

**Accuracy:** 100%
