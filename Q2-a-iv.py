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




""" ====== Change to scalar ====== """

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

#        print("Test_RMSE  fold:",fold_i," = ",test_rmse[-1] )
#        print("Train_RMSE fold:",fold_i," = ",train_rmse[-1])
        
    all_test_RMSE  = sum(test_rmse)/10
#    all_train_RMSE = sum(train_rmse)/10

#    print("\n")
#    print("Overall Test_RMSE  = ", all_test_RMSE  )
#    print("Overall Train_RMSE = ", all_train_RMSE )
    
    return test_rmse, train_rmse, all_test_RMSE





print("\n")
""" ====== <iv> Feature Encoding ====== """
print("====== <iv> Feature Encoding ======")



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
    
        a = str(bin(i))[2:].rjust(5,'0')
#        a = a.rjust(5,'0')
            
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
    

    
test_rmse_q4     = {}
train_rmse_q4    = {}
all_test_RMSE_q4 = {}

for i in np.arange(32):
    test_rmse_i  = []
    train_rmse_i = []

    X_in = X_Very_Hot[i]
    
    test_rmse_i,train_rmse_i, all_test_RMSE_i = kfold_lin_reg(X_in,Y)
    
    name = Hot_comb[i]
    
    test_rmse_q4[name]     = test_rmse_i
    train_rmse_q4[name]    = train_rmse_i
    all_test_RMSE_q4[name] = all_test_RMSE_i
    
    
del test_rmse_i,train_rmse_i,all_test_RMSE_i,i,X_in
    
print("combination with the smallest error:",min(all_test_RMSE_q4, key=all_test_RMSE_q4.get))








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

H0 = X0
H1 = X_Hot1
H2 = X_Hot2
H3 = X3
H4 = X_Hot4

X_Very_Hot_best = np.concatenate((H0,H1,H2,H3,H4),axis =1)



model_1 = linear_model.LinearRegression()
model_1.fit(X_Very_Hot_best,Y)
Y_pred_1  = model_1.predict(X_Very_Hot_best)




plt.figure()
yy = Y_pred_1
plt.title("iv feature encoding")
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
