##########################################################################################################################################
################################################### Face Recognition via Eigen-Faces #####################################################
##########################################################################################################################################

########################################################## Initial Notes #################################################################
'''
1. All "alomega=1" variables were used as debuggig checkpoints. There's no use of them in the code. 

'''


############################################################# Import #####################################################################

import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import linalg as LA 

########################################################## Initialization ################################################################

print('\n')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')



############################################# Defining Array-Vector Conversion Functions #################################################

def Convert_Array_to_Vector(a):     # To convert NxN Array to (N^2)x1 Vector form
    vector = []; 
    for i in range(len(a)):
        for j in range(len(a[0])): vector.append(a[i][j]); 
    return vector

def Convert_Vector_to_Array(v,rows,cols):     # To convert (N^2)x1 Vector to NxN Array form
    array = np.zeros((rows,cols)); k=0; 
    for i in range(rows):
        for j in range(cols): array[i][j]=v[k]; k+=1; 
    return array






##################################################### Loading Training Data ###############################################################

# Extract the training faces from the training images using Neural Network HAAR classifier method. 

def Extract_Training_Faces_NN (len_images_array):
    training_images=[];             # Stores training images data
    load_Fail=[]; load_fails=0;     # Stores image indices which failed to load as HAAR could not find faces in the image.
    size_Fail=[]; size_fails=0;     # Stores image indices whose size is smaller than my-taken default size 300px x 300px

    for i in range(1, 1+len_images_array):
        
        curr_img = cv2.imread('Train\\Train Image ('+str(i)+').jpg'); 
        curr_img = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY);          # Take an image & convert to grayscale
        rows,cols = curr_img.shape; 

        faces = face_cascade.detectMultiScale(curr_img,1.5);            # Find faces in the image, represented by (x_start,y_start,width,height) tuple
        
        if(len(faces)==0): 
            load_fails+=1; load_Fail.append(i); training_images.append(training_images[-1]); continue;   # If HAAR couldn't find face, update
        else: 
            (x0,y0,w,h)=faces[0];   xm=x0+(w/2);   ym=y0+(h/2);         # Taking midpoint of face to crop the image in 300x300

        # Cropping via slicing (Also taking care that out of bounds do not occur)
        cropped_face = curr_img[ max(0,int(ym-150)) : min(rows,int(ym+150)) , max(0,int(xm-150)) : min(cols,int(xm+150)) ]; 

        if(cropped_face.shape != (300,300)): 
            size_fails+=1; size_Fail.append(i+1); training_images.append(training_images[-1]); continue;  # Updating corresponding data
        else: 
            training_images.append(cropped_face);       # Else, add the training face into list
        
        # fig=plt.figure(); plt.imshow(curr_img, cmap='gray'); plt.show(); 
        # fig=plt.figure(); plt.imshow(cropped_face, cmap='gray'); plt.show(); 

    return training_images, load_Fail, load_fails, size_Fail, size_fails

len_training_images=319; 

training_images,load_Fail,load_fails,size_Fail,size_fails = Extract_Training_Faces_NN(len_training_images)

row = len(training_images[0]); column = len(training_images[0][0]);     # No.of rows and cols in each input image

print(load_fails,'number of images failed to load as HAAR could not find face in them. The corresponding training images are as follows:')
print(load_Fail,'\n')
print(size_fails,'number of images failed due to smaller size than 300x300 px. The corresponding training images are as follows:')
print(size_Fail,'\n')






########################################################## Average Image #################################################################

# This function finds average image
def Averageize_Img (training_images, len_training_images):
    Avg_Img = np.zeros((row,column)); 
    for i in range(len_training_images): Avg_Img = Avg_Img + training_images[i];        # Summing up all
    for i in range(row):
        for j in range(column): Avg_Img[i][j] = int( round( Avg_Img[i][j]/len_training_images ))        # Divide to find average
    return Avg_Img

Avg_Img = Averageize_Img (training_images, len_training_images)
fig = plt.figure()
plt.imshow(Avg_Img, cmap='gray')
plt.show()
# alomega=1





####################################################### Obtaining 'A' Matrix ################################################################

# This function gives A matrix which will be used to make L matrix
def Get_A_Matrix (training_images, row, column, Avg_Img):

    difference_images=[]; 
    for i in range(len_training_images):    # Finding difference images
        img = np.zeros((row,column)); img = (training_images[i]-Avg_Img); difference_images.append(img); 

    A=[];   # c=0; print('get A coeff matrix counter',end=' ')
    for array in difference_images:
        vector=Convert_Array_to_Vector(array); A.append(vector); 
        #c+=1; print(c,end=' ')
    # print('\n Reached Here \n')
    return np.array(A), np.array(difference_images)

A, diff_face_vect = Get_A_Matrix (training_images,row,column,Avg_Img)
A_T=A.T                                                                         # Transpose of A
L_matrix = np.matmul(A,A_T)                                                     # Calc L matrix
len_diff_face_vect = len_training_images 

# print('Obtained L_matrix, & now we"ll start taking k_greatest_eigens\n')
# alomega=1





################################################## Getting Eigen Values and Vectors ##########################################################

# This function gives k largest eigen values and their corresponding vector
def Get_K_Largest_EigValVec(L_matrix, A_T, k):

    # print("Calculating Eigen values and vectors")
    eigen_values,eigen_vectors_v = LA.eig(L_matrix);        # Using linear algebra to find eigens (Note: here eig-vectors are that of L matrix)
    len_eigen_values = len(eigen_values); 

    # Now, combine the eigen values with their eigen vectors in a tuple in a dictionary
    eig_val_vec={}; 
    for i in range(len_eigen_values): eig_val_vec.update({eigen_values[i]:eigen_vectors_v[:,i]})

    rev_eig_val_vec = sorted(eig_val_vec, reverse=True);      # Decreasing sort according to eigen values
    rev_sorted_k_eigs={}; counter=0; 
    
    # Convert above eig-vectors of L matrix, v's, into eig-vectors of C matrix, u's
    for i in rev_eig_val_vec:
        u = (np.matmul(A_T, eig_val_vec[i]))                  # One eigen vector u
        u = u / (np.linalg.norm(u))                           # Divide by norm as eig-vect are unit vector 
        rev_sorted_k_eigs.update({i:u}); counter+=1
        if(counter==k): break
    
    return rev_sorted_k_eigs

k=49    # To select k largest eig-val-vec to form basis for face space
rev_sorted_val_vec_dict = Get_K_Largest_EigValVec (L_matrix, A_T, k)
len_rev_sorted_val_vec_dict = k

# print('getting eigen face vectors\n')
# alomega=1






#################################################### Getting Eigen-Face Vectors ##############################################################

# Following function extracts the eigen-face-vectors from key-value pair and transforma into vectors from arrays
def Get_K_EigFaceVec(rev_sorted_val_vec_dict):
    
    eigen_faces_vectors=[]; eigen_faces_arrays=[];      # Eigen faces in vector and array forms
    temp = list(rev_sorted_val_vec_dict.items())        # Only extracting the values from key-value pairs

    for i in range(len_rev_sorted_val_vec_dict):
        temp_vec = list(temp[i][1]); eigen_faces_vectors.append(temp_vec);      # Converting to vector from numpy array
        temp_arr = Convert_Vector_to_Array(temp_vec,row,column); eigen_faces_arrays.append(temp_arr);   # Converting to array from vector

    fig = plt.figure()
    fig.subplots_adjust( hspace=0.1, wspace=0.1 )
    for i in range(1,1+k): axis = fig.add_subplot(7,7,i); axis.imshow(eigen_faces_arrays[i-1], cmap='gray'); 
    plt.show()

    return np.array(eigen_faces_vectors)

eigen_faces_vectors = Get_K_EigFaceVec (rev_sorted_val_vec_dict); 
len_eigen_faces_vectors = k

# print('Setting up Testing protocols\n')
# alomega=1








############################################### Generating Testing Protocols (Verifiers) ####################################################

# This function does as named. Makes face_space of 'total_persons' (here 20), and also makes dictionary for test_person's identity, To check and evaluate
def Set_up_Testing_Protocols(eigen_faces_vectors,diff_face_vect,k): 

    # Calculating entries of face space and append it into it
    face_space=[];      # Later, we can access as (person index = list index + 1) and corresponding entry shows weight pattern for that person



    # Here, I have clearly shown how to calculate id of first person. 
    # Other 19 are enveloped in "if(1)" statement, so that we can collapse it for better view
    Avg_weight = np.zeros(k); Counter=0;    # Call these "Utilities" for reference

    for i in range(1,17):       # The numbers 1 & 17 represent that images indexed (1 to 16) are that of first person, as this is first "for loop"
        weight = (np.matmul(eigen_faces_vectors, np.array(Convert_Array_to_Vector(diff_face_vect[i-1])))).T;    # Calcing weights
        Avg_weight = Avg_weight + weight ;                                                                      # Summing up for avg
        Counter+=1; 
    
    Avg_weight = Avg_weight / Counter;              # Averaging out weights
    face_space.append(Avg_weight); 
    Avg_weight = np.zeros(k); Counter=0;            # Set "Utilities" to default

    if(1):      # This "if(1)" block contains above calculations for all other 19 personas, but in a compressed form (for less scrolling). 
        for i in range(17,32): weight = (np.matmul(eigen_faces_vectors, np.array(Convert_Array_to_Vector(diff_face_vect[i-1])))).T; Avg_weight = Avg_weight + weight ; Counter+=1; 
        Avg_weight=Avg_weight/Counter; face_space.append(Avg_weight); Avg_weight = np.zeros(k); Counter=0; 
        for i in range(32,49): weight = (np.matmul(eigen_faces_vectors, np.array(Convert_Array_to_Vector(diff_face_vect[i-1])))).T; Avg_weight = Avg_weight + weight ; Counter+=1; 
        Avg_weight=Avg_weight/Counter; face_space.append(Avg_weight); Avg_weight = np.zeros(k); Counter=0; 
        for i in range(49,65): weight = (np.matmul(eigen_faces_vectors, np.array(Convert_Array_to_Vector(diff_face_vect[i-1])))).T; Avg_weight = Avg_weight + weight ; Counter+=1; 
        Avg_weight=Avg_weight/Counter; face_space.append(Avg_weight); Avg_weight = np.zeros(k); Counter=0; 
        for i in range(65,83): weight = (np.matmul(eigen_faces_vectors, np.array(Convert_Array_to_Vector(diff_face_vect[i-1])))).T; Avg_weight = Avg_weight + weight ; Counter+=1; 
        Avg_weight=Avg_weight/Counter; face_space.append(Avg_weight); Avg_weight = np.zeros(k); Counter=0; 
        for i in range(83,98): weight = (np.matmul(eigen_faces_vectors, np.array(Convert_Array_to_Vector(diff_face_vect[i-1])))).T; Avg_weight = Avg_weight + weight ; Counter+=1; 
        Avg_weight=Avg_weight/Counter; face_space.append(Avg_weight); Avg_weight = np.zeros(k); Counter=0; 
        for i in range(98,114): weight = (np.matmul(eigen_faces_vectors, np.array(Convert_Array_to_Vector(diff_face_vect[i-1])))).T; Avg_weight = Avg_weight + weight ; Counter+=1; 
        Avg_weight=Avg_weight/Counter; face_space.append(Avg_weight); Avg_weight = np.zeros(k); Counter=0; 
        for i in range(114,120): weight = (np.matmul(eigen_faces_vectors, np.array(Convert_Array_to_Vector(diff_face_vect[i-1])))).T; Avg_weight = Avg_weight + weight ; Counter+=1; 
        Avg_weight=Avg_weight/Counter; face_space.append(Avg_weight); Avg_weight = np.zeros(k); Counter=0; 
        for i in range(120,135): weight = (np.matmul(eigen_faces_vectors, np.array(Convert_Array_to_Vector(diff_face_vect[i-1])))).T; Avg_weight = Avg_weight + weight ; Counter+=1; 
        Avg_weight=Avg_weight/Counter; face_space.append(Avg_weight); Avg_weight = np.zeros(k); Counter=0; 
        for i in range(135,151): weight = (np.matmul(eigen_faces_vectors, np.array(Convert_Array_to_Vector(diff_face_vect[i-1])))).T; Avg_weight = Avg_weight + weight ; Counter+=1; 
        Avg_weight=Avg_weight/Counter; face_space.append(Avg_weight); Avg_weight = np.zeros(k); Counter=0;  
        for i in range(151,171): weight = (np.matmul(eigen_faces_vectors, np.array(Convert_Array_to_Vector(diff_face_vect[i-1])))).T; Avg_weight = Avg_weight + weight ; Counter+=1; 
        Avg_weight=Avg_weight/Counter; face_space.append(Avg_weight); Avg_weight = np.zeros(k); Counter=0;  
        for i in range(171,188): weight = (np.matmul(eigen_faces_vectors, np.array(Convert_Array_to_Vector(diff_face_vect[i-1])))).T; Avg_weight = Avg_weight + weight ; Counter+=1; 
        Avg_weight=Avg_weight/Counter; face_space.append(Avg_weight); Avg_weight = np.zeros(k); Counter=0;  
        for i in range(188,203): weight = (np.matmul(eigen_faces_vectors, np.array(Convert_Array_to_Vector(diff_face_vect[i-1])))).T; Avg_weight = Avg_weight + weight ; Counter+=1; 
        Avg_weight=Avg_weight/Counter; face_space.append(Avg_weight); Avg_weight = np.zeros(k); Counter=0;  
        for i in range(203,218): weight = (np.matmul(eigen_faces_vectors, np.array(Convert_Array_to_Vector(diff_face_vect[i-1])))).T; Avg_weight = Avg_weight + weight ; Counter+=1; 
        Avg_weight=Avg_weight/Counter; face_space.append(Avg_weight); Avg_weight = np.zeros(k); Counter=0;  
        for i in range(218,234): weight = (np.matmul(eigen_faces_vectors, np.array(Convert_Array_to_Vector(diff_face_vect[i-1])))).T; Avg_weight = Avg_weight + weight ; Counter+=1; 
        Avg_weight=Avg_weight/Counter; face_space.append(Avg_weight); Avg_weight = np.zeros(k); Counter=0;  
        for i in range(243,258): weight = (np.matmul(eigen_faces_vectors, np.array(Convert_Array_to_Vector(diff_face_vect[i-1])))).T; Avg_weight = Avg_weight + weight ; Counter+=1; 
        Avg_weight=Avg_weight/Counter; face_space.append(Avg_weight); Avg_weight = np.zeros(k); Counter=0;  
        for i in range(258,272): weight = (np.matmul(eigen_faces_vectors, np.array(Convert_Array_to_Vector(diff_face_vect[i-1])))).T; Avg_weight = Avg_weight + weight ; Counter+=1; 
        Avg_weight=Avg_weight/Counter; face_space.append(Avg_weight); Avg_weight = np.zeros(k); Counter=0;  
        for i in range(272,288): weight = (np.matmul(eigen_faces_vectors, np.array(Convert_Array_to_Vector(diff_face_vect[i-1])))).T; Avg_weight = Avg_weight + weight ; Counter+=1; 
        Avg_weight=Avg_weight/Counter; face_space.append(Avg_weight); Avg_weight = np.zeros(k); Counter=0;  
        for i in range(288,302): weight = (np.matmul(eigen_faces_vectors, np.array(Convert_Array_to_Vector(diff_face_vect[i-1])))).T; Avg_weight = Avg_weight + weight ; Counter+=1; 
        Avg_weight=Avg_weight/Counter; face_space.append(Avg_weight); Avg_weight = np.zeros(k); Counter=0;  
        for i in range(302,320): weight = (np.matmul(eigen_faces_vectors, np.array(Convert_Array_to_Vector(diff_face_vect[i-1])))).T; Avg_weight = Avg_weight + weight ; Counter+=1; 
        Avg_weight=Avg_weight/Counter; face_space.append(Avg_weight); Avg_weight = np.zeros(k); Counter=0;  




    # Creating a dictionary that tells which test image is of which person's index
    test_persona={};    # (key : value) represent (test_image_index_i : person_index from dataset)
    for i in range(1,6): test_persona[i]=1;      # test_images indexed (1 to 5) are that of first person's, as this is first for loop. 

    if(1):      # This "if(1)" block contains similar loops for all other 19 personas. 
        for i in range(6,11): test_persona[i]=2; 
        for i in range(11,16): test_persona[i]=3; 
        for i in range(16,21): test_persona[i]=4; 
        for i in range(21,26): test_persona[i]=5; 
        for i in range(26,31): test_persona[i]=6; 
        for i in range(31,36): test_persona[i]=7; 
        for i in range(36,37): test_persona[i]=8; 
        for i in range(37,42): test_persona[i]=9; 
        for i in range(42,47): test_persona[i]=10; 
        for i in range(47,52): test_persona[i]=11; 
        for i in range(52,57): test_persona[i]=12; 
        for i in range(57,61): test_persona[i]=13; 
        for i in range(61,66): test_persona[i]=14; 
        for i in range(66,70): test_persona[i]=15; 
        for i in range(70,75): test_persona[i]=16; 
        for i in range(75,81): test_persona[i]=17; 
        for i in range(81,87): test_persona[i]=18; 
        for i in range(87,93): test_persona[i]=19; 
        for i in range(93,97): test_persona[i]=20; 

    return face_space, test_persona

total_persons=20; 
face_space, test_persona = Set_up_Testing_Protocols (eigen_faces_vectors, diff_face_vect,k); 

# print('\nNow load the testing dataset \n')
# alomega=1







##################################################### Generating Test Data ###################################################################

# Extract the testing faces from the testing images using Neural Network HAAR classifier method. 
def Extract_Testing_Faces_NN (len_images_array):

    testing_images=[];              # Stores testing images data
    Load_Fail=[]; Load_fails=0;     # Stores image indices which failed to load as HAAR could not find faces in the image.
    Size_Fail=[]; Size_fails=0;     # Stores image indices whose size is smaller than my-taken default size 300px x 300px

    for i in range(1, 1+len_images_array):
        
        curr_img = cv2.imread('Test\\Test Image ('+str(i)+').jpg')
        curr_img = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)       # Take an image & convert to grayscale
        rows,cols = curr_img.shape

        faces = face_cascade.detectMultiScale(curr_img,1.5)         # Find faces in the image, represented by (x_start,y_start,width,height) tuple
            
        if (len(faces)==0): 
            Load_fails+=1; Load_Fail.append(i); testing_images.append(testing_images[-1]); continue;   # If HAAR couldn't find face, update
        else: 
            (x0,y0,w,h)=faces[0];   xm=x0+(w/2);   ym=y0+(h/2);         # Taking midpoint of face to crop the image in 300x300

        # Cropping via slicing (Also taking care that out of bounds do not occur)
        cropped_face = curr_img[ max(0,int(ym-150)) : min(rows,int(ym+150)) , max(0,int(xm-150)) : min(cols,int(xm+150)) ]; 

        if(cropped_face.shape != (300,300)): 
            Size_fails+=1; Size_Fail.append(i+1); testing_images.append(testing_images[-1]); continue;  # Updating corresponding data
        else: 
            testing_images.append(cropped_face);       # Else, add the training face into list
                
        # fig=plt.figure(); plt.imshow(curr_img, cmap='gray'); plt.show(); 
        # fig=plt.figure(); plt.imshow(cropped_face, cmap='gray'); plt.show(); 
    
    return testing_images, Load_Fail, Load_fails, Size_Fail, Size_fails

len_testing_images = 96
testing_images, Load_Fail, Load_fails, Size_Fail, Size_fails = Extract_Testing_Faces_NN(len_testing_images)

print()
print(Load_fails,'number of images failed to load as HAAR could not find face in them. The corresponding testing images are as follows:')
print(Load_Fail,'\n')
print(Size_fails,'number of images failed due to smaller size than 300x300 px. The corresponding testing images are as follows:')
print(Size_Fail,'\n')

# print('\n Now start the actual testing')
# alomega=1






######################################################## Testing the Model ###################################################################

# Now, this function will actually test the model/algorithm/dataset. 
def Test_the_model (testing_images, Avg_Img, eigen_faces_vectors, total_persons, face_space, test_persona):

    curr_test_img_index = correct = incorrect = notface = 0; results=[];      # Counts for keeping record/track for evaluation
    print("Now, results will be printed for each of the test image. ")

    for new_face_img in testing_images:

        curr_test_img_index+=1
        
        weights = (np.matmul(eigen_faces_vectors, np.array(Convert_Array_to_Vector( new_face_img - Avg_Img )))).T   # Calcing weights for this new face image


        min_epsilon_k = 10**7; estimated_person = 0;        # To keep min_epsi value and estimated_person which is estimated by the algo
        for k in range(total_persons):                      # For each distinct person_k in face space, 

            epsilon_k = LA.norm (weights-face_space[k]);    # Finding norm and save it to epsilon

            if (min_epsilon_k > epsilon_k): 
                min_epsilon_k = epsilon_k; estimated_person = k+1 ;     # Update minimum epsilons
        
        if    (min_epsilon_k > 25000): notface+=1; results.append(-1)       # For non-faces, value 25000 obtained from experiments. Still it is sensitive i.e. may change
        elif  (estimated_person == test_persona[curr_test_img_index]): correct+=1; results.append(1);   # If estimated person is the correct person as from dictionary
        else: incorrect+=1; results.append(0); 

        ### Printing work. 
        print('Current Test_Image index =',curr_test_img_index)
        print('Estimated Person Index =',estimated_person)
        print('Actual Person Index =',test_persona[curr_test_img_index])
        
        if    (min_epsilon_k > 25000):  print("Not a face image, most probably.")
        elif  (estimated_person == test_persona[curr_test_img_index]): print("Algorithm correctly identified this person.")
        else: print("Algorithm could not identify this person correctly.")

        print('Until now, Total Correct =',correct,', Total Incorect =',incorrect,', Non-Face =',notface)
        # print('min_epsilon_k =',min_epsilon_k)
        print('')
        # alomega=1

    return results,correct,incorrect,notface

results,correct,incorrect,notface = Test_the_model(testing_images, Avg_Img, eigen_faces_vectors, total_persons, face_space, test_persona)




############################################################# Final Results ################################################################

print('\n FINAL RESULTS :\n')
print('We have chosen',k,'eigen faces to evaluate this.')
print('Possible Non-face results are '+str(notface)+'.')
print('Correct Predictions by the Algorithm:',correct)
print('Incorrect Predictions by the Algorithm:',incorrect)
print('Accurcacy:', round(float((correct/(correct+incorrect))*100),2),'%')

print('\n Program Terminated \n')

exitcode=input("Press Enter to Exit.")

##########################################################################################################################################