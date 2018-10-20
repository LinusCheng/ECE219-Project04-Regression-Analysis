import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import KFold
#from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn import metrics
#from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
#from sklearn.feature_selection import f_regression, mutual_info_regression
import matplotlib.pyplot as plt
import time



""" ====== Prepare data ====== """

content = pd.read_csv("network_backup_dataset.csv")
content = content.replace({'Day of Week': {'Monday' : 0, 'Tuesday' : 1, 'Wednesday' : 2 ,
                                         'Thursday' : 3, 'Friday' : 4,
                                         'Saturday' : 5, 'Sunday' : 6 }})

WorkFlow_list = sorted(pd.unique(content['Work-Flow-ID']))
File_list     = sorted(pd.unique(content['File Name']))  
num_WorkFlow  = len(WorkFlow_list)
num_File      = len(File_list)

for i in np.arange(num_WorkFlow):
    content = content.replace({'Work-Flow-ID': {'work_flow_%d'%i:i}})

for i in np.arange(num_File):
    content = content.replace({'File Name': {'File_%d'%i:i}})


content_all_np = content.values
X = content_all_np[: , 0:5]
Y = content_all_np[: ,  5 ]


del WorkFlow_list,File_list,content_all_np, num_WorkFlow,num_File,i
# delete unused variables

def hot32comb(X):

    # the 32 combinations, split X into 5 columns
    enc1 =  preprocessing.OneHotEncoder(sparse = False)
    num_sample = X.shape[0]
    
    X0 = X[:,0].reshape((num_sample,1))
    X1 = X[:,1].reshape((num_sample,1))
    X2 = X[:,2].reshape((num_sample,1))
    X3 = X[:,3].reshape((num_sample,1))
    X4 = X[:,4].reshape((num_sample,1))
    
    X_Hot0 = enc1.fit_transform(X0)
    X_Hot1 = enc1.fit_transform(X1)
    X_Hot2 = enc1.fit_transform(X2)
    X_Hot3 = enc1.fit_transform(X3)
    X_Hot4 = enc1.fit_transform(X4)
    
    X_Very_Hot = []
    Hot_comb   = []
    
    for i in np.arange(32):
    
        a = str(bin(i))[2:]
        a = a.rjust(5,'0')
            
        H0 = X0*int(a[0]) + X_Hot0*(1-int(a[0]))
        H1 = X1*int(a[1]) + X_Hot1*(1-int(a[1]))
        H2 = X2*int(a[2]) + X_Hot2*(1-int(a[2]))
        H3 = X3*int(a[3]) + X_Hot3*(1-int(a[3]))
        H4 = X4*int(a[4]) + X_Hot4*(1-int(a[4]))  
        
        X_Very_Hot_i = np.concatenate((H0,H1,H2,H3,H4),axis =1)
        X_Very_Hot.append(X_Very_Hot_i)
        Hot_comb.append(a)
    
    print("X_Very_Hot generated")
    return X_Very_Hot,Hot_comb


X_Very_Hot,Hot_comb = hot32comb(X)



print("\n")
""" ====== <v> Control ill-conditioning & over-fiting ====== """
print("==== <v> Control ill-condi & over-fit ====")


def kfold_ridg(X,Y,alpha):

    test_rmse  = []
    train_rmse = []

    kf = KFold(n_splits=10, random_state=None, shuffle=False)
    fold_i =0
    for train_id, test_id in kf.split(X):
        fold_i += 1
        X_train, X_test = X[train_id], X[test_id]
        Y_train, Y_test = Y[train_id], Y[test_id]

        model_1 = linear_model.Ridge(alpha)
        model_1.fit(X_train,Y_train)

        Y_test_pred  = model_1.predict(X_test)
        Y_train_pred = model_1.predict(X_train)

        test_rmse.append ( np.sqrt( metrics.mean_squared_error(Y_test , Y_test_pred )))
        train_rmse.append( np.sqrt( metrics.mean_squared_error(Y_train, Y_train_pred)))

    all_test_RMSE  = sum(test_rmse)/10
    
    return test_rmse, train_rmse, all_test_RMSE



def kfold_lass(X,Y,alpha):

    test_rmse  = []
    train_rmse = []

    kf = KFold(n_splits=10, random_state=None, shuffle=False)
    fold_i =0
    for train_id, test_id in kf.split(X):
        fold_i += 1
        X_train, X_test = X[train_id], X[test_id]
        Y_train, Y_test = Y[train_id], Y[test_id]

        model_1 = linear_model.Lasso(alpha)
        model_1.fit(X_train,Y_train)

        Y_test_pred  = model_1.predict(X_test)
        Y_train_pred = model_1.predict(X_train)

        test_rmse.append ( np.sqrt( metrics.mean_squared_error(Y_test , Y_test_pred )))
        train_rmse.append( np.sqrt( metrics.mean_squared_error(Y_train, Y_train_pred)))

    all_test_RMSE  = sum(test_rmse)/10
    
    return test_rmse, train_rmse, all_test_RMSE



def kfold_elst(X,Y,alpha,l1_ratio):

    test_rmse  = []
    train_rmse = []

    kf = KFold(n_splits=10, random_state=None, shuffle=False)
    fold_i =0
    for train_id, test_id in kf.split(X):
        fold_i += 1
        X_train, X_test = X[train_id], X[test_id]
        Y_train, Y_test = Y[train_id], Y[test_id]

        model_1 = linear_model.ElasticNet(alpha,l1_ratio)
        model_1.fit(X_train,Y_train)

        Y_test_pred  = model_1.predict(X_test)
        Y_train_pred = model_1.predict(X_train)

        test_rmse.append ( np.sqrt( metrics.mean_squared_error(Y_test , Y_test_pred )))
        train_rmse.append( np.sqrt( metrics.mean_squared_error(Y_train, Y_train_pred)))

    all_test_RMSE  = sum(test_rmse)/10
    
    return test_rmse, train_rmse, all_test_RMSE



def hot32acc_best(X_Very_Hot,Hot_comb,Y,reg_model,alpha,l1_ratio):
    all_test_RMSE_i = 0
    best_RMSE       = 0
    best_comb       = 0
    for i in np.arange(32):
        X_in = X_Very_Hot[i]
        if reg_model == 'ridg':
            _ ,_ , all_test_RMSE_i = kfold_ridg(X_in,Y,alpha)
        elif reg_model == 'lass':
            _ ,_ , all_test_RMSE_i = kfold_lass(X_in,Y,alpha)
        elif reg_model == 'elst':
            _ ,_ , all_test_RMSE_i = kfold_elst(X_in,Y,alpha,l1_ratio)
        else:
            print("Wrong Input")
            break
        
        name = Hot_comb[i]
        
        if i ==0:
            best_RMSE = all_test_RMSE_i
            best_comb = name
        
        if all_test_RMSE_i < best_RMSE:
            best_RMSE = all_test_RMSE_i
            best_comb = name

    return best_RMSE,best_comb


print("\n")
""" === 1. Ridge === """
print("=== 1. Ridge ===")
l1_ratio =0

alpha_all = np.array([0.1,0.3,0.5,1,3,5,10,20,40,80,120])


RMSE_ridge = np.zeros((len(alpha_all),3))


i=0
for alpha in alpha_all:

    best_RMSE,best_comb = hot32acc_best(X_Very_Hot,Hot_comb,Y,'ridg',alpha,l1_ratio)

    RMSE_ridge[i,0] = alpha
    RMSE_ridge[i,1] = best_comb
    RMSE_ridge[i,2] = best_RMSE
    i += 1


print("RMSE_ridge generated")

RMSE_ridg_column = RMSE_ridge[:,2]
RMSE_ind         = np.argmin(RMSE_ridg_column)

#alpha = 5
#0.08836773802910462




print("\n")
""" === 2. Lasso === """
print("=== 2. Lasso ===")
l1_ratio =1

#alpha_all = np.array([0.001,0.002,0.005,0.01,0.1,0.3,0.5,1,5,10])

#alpha_all = np.array([0.0001])

alpha_all = np.array([0.0001,0.001,0.002,0.005,0.01])

RMSE_lasso = np.zeros((len(alpha_all),3))


i=0
for alpha in alpha_all:

    best_RMSE,best_comb = hot32acc_best(X_Very_Hot,Hot_comb,Y,'lass',alpha,l1_ratio)

    RMSE_lasso[i,0] = alpha
    RMSE_lasso[i,1] = best_comb
    RMSE_lasso[i,2] = best_RMSE
    i += 1


print("RMSE_lasso generated")


RMSE_lass_column = RMSE_lasso[:,2]
RMSE_ind         = np.argmin(RMSE_lass_column)


print("min alpha index is" , RMSE_ind)
print("The error was ",RMSE_lass_column[RMSE_ind])


#alpha = 0.0001
#The error was  0.08837160165492905


print("\n")
""" === 3. Elastic === """
print("=== 3. Elastic ===")



alpha_all     = np.array([0.5,5.0,20.0,80.0,120.0,200.0])
l1_ratio_all  = np.array([0.01,0.1,0.3,0.5,0.7,0.9,0.99])


RMSE_elst_list = np.zeros((len(alpha_all)*len(l1_ratio_all),3))

RMSE_elst_map  = np.zeros((len(alpha_all)+1, len(l1_ratio_all)+1 ))
RMSE_elst_map[0,1:] = l1_ratio_all #.reshape(1,len(l1_ratio_all))
RMSE_elst_map[1:,0] = alpha_all #.reshape(len(alpha_all),1)


i=0
k=0
for alpha in alpha_all:
    j=0
    t1 = time.time()
    for l1_ratio in l1_ratio_all:
        best_RMSE,best_comb = hot32acc_best(X_Very_Hot,Hot_comb,Y,'elst',alpha,l1_ratio)
        RMSE_elst_list[k,0] = alpha
        RMSE_elst_list[k,1] = best_comb
        RMSE_elst_list[k,2] = best_RMSE
        
        RMSE_elst_map[i+1,j+1]  = best_RMSE
        j += 1
        k += 1
    t2 = time.time()
    print("alpha = ",alpha," done")
    print("Time consumed:",t2-t1)
    i +=1
    
print("RMSE_elst generated")


# Find the minimum error in the result
#print("Finding the minimum error in the result")


RMSE_column = RMSE_elst_list[:,2]
RMSE_ind    = np.argmin(RMSE_column)
best_alpha  = RMSE_elst_list[RMSE_ind,0]
best_comb   = str(int(RMSE_elst_list[RMSE_ind,1])).rjust(5,'0')
best_error  = RMSE_elst_list[RMSE_ind,2]

print("Best error = ",best_error,"\nalpha = ",best_alpha,"\ncombination = ",best_comb)

#Best error =  0.10021045142290859 
#alpha =  0.5 
#combination =  01100
#l1_ratio = 0.01






print("\n\n")
""" ====== plotting predicted vs true value ====== """
print("====== plot pred vs true output ======")


enc1 =  preprocessing.OneHotEncoder(sparse = False)
num_sample = X.shape[0]

X0 = X[:,0].reshape((num_sample,1))
X1 = X[:,1].reshape((num_sample,1))
X2 = X[:,2].reshape((num_sample,1))
X3 = X[:,3].reshape((num_sample,1))
X4 = X[:,4].reshape((num_sample,1))
X_Hot0 = enc1.fit_transform(X0)
X_Hot1 = enc1.fit_transform(X1)
X_Hot2 = enc1.fit_transform(X2)
X_Hot3 = enc1.fit_transform(X3)
X_Hot4 = enc1.fit_transform(X4)





##############
H0 = X0
H1 = X_Hot1
H2 = X_Hot2
H3 = X_Hot3
H4 = X4
alpha = 5

X_Very_Hot_best = np.concatenate((H0,H1,H2,H3,H4),axis =1)
model_1 = linear_model.Ridge(alpha)
model_1.fit(X_Very_Hot_best,Y)
Y_pred_1  = model_1.predict(X_Very_Hot_best)



plt.figure()
yy = Y_pred_1
plt.title("v ridge")
xx = Y
plt.scatter(xx, yy, s = 1,  alpha=0.01)
plt.xlabel('true value')
plt.ylabel('fitted value')
xi = range(0,2)
yi = [i for i in xi]
plt.axis([-0.03,1.03,-0.03,1.03])
plt.plot(xi,yi,color='red')

plt.figure()
xx = Y_pred_1
yy = np.abs(xx-Y)
plt.scatter(xx, yy, s=1, alpha=0.01)
plt.axis([-0.03,1.03,-0.03,1.03])
plt.xlabel('fitted value')
plt.ylabel('residuals')



##############
H0 = X_Hot0
H1 = X_Hot1
H2 = X_Hot2
H3 = X_Hot3
H4 = X_Hot4
alpha = 0.0001
X_Very_Hot_best = np.concatenate((H0,H1,H2,H3,H4),axis =1)
model_2 = linear_model.Lasso(alpha)
model_2.fit(X_Very_Hot_best,Y)
Y_pred_2  = model_2.predict(X_Very_Hot_best)



plt.figure()
yy = Y_pred_2
plt.title("v lasso")
xx = Y
plt.scatter(xx, yy, s = 1,  alpha=0.01)
plt.xlabel('true value')
plt.ylabel('fitted value')
xi = range(0,2)
yi = [i for i in xi]
plt.axis([-0.03,1.03,-0.03,1.03])
plt.plot(xi,yi,color='red')

plt.figure()
xx = Y_pred_2
yy = np.abs(xx-Y)
plt.scatter(xx, yy, s=1, alpha=0.01)
plt.axis([-0.03,1.03,-0.03,1.03])
plt.xlabel('fitted value')
plt.ylabel('residuals')



##############
H0 = X_Hot0
H1 = X1
H2 = X2
H3 = X_Hot3
H4 = X_Hot4
alpha = 0.5
l1_ratio = 0.01
X_Very_Hot_best = np.concatenate((H0,H1,H2,H3,H4),axis =1)
model_3 = linear_model.ElasticNet(alpha,l1_ratio)
model_3.fit(X_Very_Hot_best,Y)
Y_pred_3  = model_3.predict(X_Very_Hot_best)

plt.figure()
yy = Y_pred_3
plt.title("v elastic")
xx = Y
plt.scatter(xx, yy, s = 1,  alpha=0.01)
plt.xlabel('true value')
plt.ylabel('fitted value')
xi = range(0,2)
yi = [i for i in xi]
plt.axis([-0.03,1.03,-0.03,1.03])
plt.plot(xi,yi,color='red')

plt.figure()
xx = Y_pred_3
yy = np.abs(xx-Y)
plt.scatter(xx, yy, s=1, alpha=0.01)
plt.axis([-0.03,1.03,-0.03,1.03])
plt.xlabel('fitted value')
plt.ylabel('residuals')

print("completed")
