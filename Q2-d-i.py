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


cont_np = content.values

X_list = []
Y_list = []

for i in np.arange(5):
    cont_list_i = cont_np[np.where(cont_np[:,3] == i)]
    X_i = np.delete(cont_list_i[: , 0:5],3,axis=1)
    Y_i = cont_list_i[: ,  5 ]
    X_list.append(X_i)
    Y_list.append(Y_i)



#X_org = content_all_np[: , 0:5]
#Y_org = content_all_np[: ,  5 ]


del WorkFlow_list,File_list,cont_np, num_WorkFlow,num_File,i,X_i,Y_i,cont_list_i
# delete unused variables





def hot16comb(X):

    # the 16 combinations, split X into 5 columns
    enc1 =  preprocessing.OneHotEncoder(sparse = False)
    num_sample = X.shape[0]
    
    X0 = X[:,0].reshape((num_sample,1))
    X1 = X[:,1].reshape((num_sample,1))
    X2 = X[:,2].reshape((num_sample,1))
    X3 = X[:,3].reshape((num_sample,1))

    
    X_Hot0 = enc1.fit_transform(X0)
    X_Hot1 = enc1.fit_transform(X1)
    X_Hot2 = enc1.fit_transform(X2)
    X_Hot3 = enc1.fit_transform(X3)
    
    X_Very_Hot = []
    Hot_comb   = []
    
    for i in np.arange(16):
    
        a = str(bin(i))[2:].rjust(4,'0')
            
        H0 = X0*int(a[0]) + X_Hot0*(1-int(a[0]))
        H1 = X1*int(a[1]) + X_Hot1*(1-int(a[1]))
        H2 = X2*int(a[2]) + X_Hot2*(1-int(a[2]))
        H3 = X3*int(a[3]) + X_Hot3*(1-int(a[3]))
        
        X_Very_Hot_i = np.concatenate((H0,H1,H2,H3),axis =1)
        X_Very_Hot.append(X_Very_Hot_i)
        Hot_comb.append(a)
    
    print("X_Very_Hot generated")
    return X_Very_Hot,Hot_comb


X_Very_Hot = []
Hot_comb   = []
for i in np.arange(5):

    X_Very_Hot_i,Hot_comb_i = hot16comb(X_list[i])
    X_Very_Hot.append(X_Very_Hot_i)
    Hot_comb.append(Hot_comb_i)

del X_Very_Hot_i,Hot_comb_i


print("\n")
""" ====== <i> linear regression ====== """
print("====== <i> linear regression ======")


def kfold_lin_reg(X,Y):
    test_rmse  = []
    train_rmse = []
    kf = KFold(n_splits=10, random_state=None, shuffle=False)
    fold_i =0
    for train_id, test_id in kf.split(X):
        fold_i += 1
        X_train, X_test = X[train_id], X[test_id]
        Y_train, Y_test = Y[train_id], Y[test_id]

        model_1 = linear_model.LinearRegression()
        model_1.fit(X_train,Y_train)

        Y_test_pred  = model_1.predict(X_test)
        Y_train_pred = model_1.predict(X_train)

        test_rmse.append ( np.sqrt( metrics.mean_squared_error(Y_test , Y_test_pred )))
        train_rmse.append( np.sqrt( metrics.mean_squared_error(Y_train, Y_train_pred)))

    avg_test_RMSE   = sum(test_rmse)/10
    avg_train_RMSE  = sum(train_rmse)/10

    return avg_test_RMSE,avg_train_RMSE



def hot16acc_best(X_Very_Hot,Hot_comb,Y,reg_model,alpha,l1_ratio):
    avg_test_RMSE_i  = 0
    avg_train_RMSE_i = 0
    best_RMSE_test   = 0
    best_RMSE_train  = 0
    best_comb        = 0
    for i in np.arange(16):
        X_in = X_Very_Hot[i]
        
        if reg_model   == 'lin_reg':
            avg_test_RMSE_i,avg_train_RMSE_i = kfold_lin_reg(X_in,Y)
#        elif reg_model == 'ridg':
#            avg_test_RMSE_i,avg_train_RMSE_i = kfold_ridg(X_in,Y,alpha)
#        elif reg_model == 'lass':
#            avg_test_RMSE_i,avg_train_RMSE_i = kfold_lass(X_in,Y,alpha)
#        elif reg_model == 'elst':
#            avg_test_RMSE_i,avg_train_RMSE_i = kfold_elst(X_in,Y,alpha,l1_ratio)
        else:
            print("Wrong Input")
            break
        
        name = Hot_comb[i]
        
        if i ==0:
            best_RMSE_test   = avg_test_RMSE_i
            best_comb        = name
            best_RMSE_train  = avg_train_RMSE_i
        
        if avg_test_RMSE_i < best_RMSE_test:
            best_RMSE_test   = avg_test_RMSE_i
            best_comb        = name
            best_RMSE_train  = avg_train_RMSE_i

        
    return best_RMSE_test,best_RMSE_train,best_comb


alpha = 5
l1_ratio =1

i =0

best_RMSE_lr_test  = {}
best_RMSE_lr_train = {}
best_comb_lr       = {}

for i in np.arange(5):
    best_RMSE_test_i,best_RMSE_train_i,best_comb_i = hot16acc_best(X_Very_Hot[i],Hot_comb[i],Y_list[i],'lin_reg',alpha,l1_ratio)
    best_RMSE_lr_test['work_flow_%d'%i]  = best_RMSE_test_i
    best_RMSE_lr_train['work_flow_%d'%i] = best_RMSE_train_i
    best_comb_lr['work_flow_%d'%i] = best_comb_i

del best_RMSE_test_i,best_RMSE_train_i,best_comb_i,i,alpha,l1_ratio





print("\n\n")
""" ====== plotting predicted vs true value ====== """
print("====== plot pred vs true output ======")


for wf in np.arange(5):

    enc1 =  preprocessing.OneHotEncoder(sparse = False)

    X = X_list[wf]
    Y = Y_list[wf]
    a = best_comb_lr['work_flow_%d'%wf]

    num_sample = X.shape[0]
    
    X0 = X[:,0].reshape((num_sample,1))
    X1 = X[:,1].reshape((num_sample,1))
    X2 = X[:,2].reshape((num_sample,1))
    X3 = X[:,3].reshape((num_sample,1))
    X_Hot0 = enc1.fit_transform(X0)
    X_Hot1 = enc1.fit_transform(X1)
    X_Hot2 = enc1.fit_transform(X2)
    X_Hot3 = enc1.fit_transform(X3)

    H0 = X0*int(a[0]) + X_Hot0*(1-int(a[0]))
    H1 = X1*int(a[1]) + X_Hot1*(1-int(a[1]))
    H2 = X2*int(a[2]) + X_Hot2*(1-int(a[2]))
    H3 = X3*int(a[3]) + X_Hot3*(1-int(a[3]))

    X_Very_Hot_best = np.concatenate((H0,H1,H2,H3),axis =1)
    model_1 = linear_model.LinearRegression()
    model_1.fit(X_Very_Hot_best,Y)
    Y_pred_1  = model_1.predict(X_Very_Hot_best)

    plt.figure()
    yy = Y_pred_1
    plt.title('work_flow_%d'%wf)
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




print("completed")

